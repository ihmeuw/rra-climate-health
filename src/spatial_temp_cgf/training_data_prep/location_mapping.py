from pathlib import Path
import geopandas as gpd
import pandas as pd


LSAE_RAW_SHAPE_PATH = Path('/snfs1/WORK/11_geospatial/admin_shapefiles/2023_10_30/lbd_standard_admin_2.shp')
LSAE_SHAPE_PATH = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/lbd_admin2.parquet")
LSAE_HIERARCHY_PATH = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/lbd_hierarchy.parquet")

FHS_RAW_SHAPE_PATH = Path('/snfs1/DATA/SHAPE_FILES/GBD_geographies/master/GBD_2021/master/shapefiles/GBD2021_analysis_final.shp')
FHS_SHAPE_PATH = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/fhs_most_detailed.parquet")
FHS_HIERARCHY_PATH = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/fhs_hierarchy.parquet")

LSAE_FHS_MAPPING_PATH = Path("/mnt/team/rapidresponse/pub/population/data/02-processed-data/ihme/lbd_admin2_fhs_mapping.parquet")

WORLDPOP_FILEPATH = Path('/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/worldpop-constrained')


def make_parquet_files():
    import db_queries
    print('Copying LSAE Shapes')
    gpd.read_file(LSAE_RAW_SHAPE_PATH).to_parquet(LSAE_SHAPE_PATH)
    print('Copying LSAE hierarchy')
    db_queries.get_location_metadata(125, release_id=16).to_parquet(LSAE_HIERARCHY_PATH)

    print('Copying FHS Shapes')
    gpd.read_file(FHS_RAW_SHAPE_PATH).to_parquet(FHS_SHAPE_PATH)
    print('Copying LSAE hierarchy')
    db_queries.get_location_metadata(30, release_id=16).to_parquet(FHS_HIERARCHY_PATH)


def build_fhs_lsae_mapping():
    lsae_shapes = gpd.read_parquet(LSAE_SHAPE_PATH)
    lsae_loc_meta = pd.read_parquet(LSAE_HIERARCHY_PATH)
    fhs_shapes = gpd.read_parquet(FHS_SHAPE_PATH)
    fhs_loc_meta = pd.read_parquet(FHS_HIERARCHY_PATH)
    
    lsae_loc_most_detailed = (
        lsae_loc_meta.query("most_detailed == 1")
        .loc[:, ['location_id', 'ihme_loc_id', 'location_name', 'path_to_top_parent']]
        .rename(columns={'location_id':'lsae_location_id', 'ihme_loc_id':'lsae_ihme_loc_id', 'location_name':'lsae_location_name'})
    )
    
    # FIX THE ONE PROBLEMATIC ETHIOPIA SUBNATIONAL
    SNNP_lsae_locid = lsae_loc_meta.query("location_name == 'Southern Nations, Nationalities, and Peoples'").location_id.item()
    SNNP_fhs_locid = fhs_loc_meta.query("location_name == 'Southern Nations, Nationalities, and Peoples'").location_id.item()
    fhs_loc_meta.loc[fhs_loc_meta.location_id == SNNP_fhs_locid, 'location_id'] = SNNP_lsae_locid
    
    fhs_loc_most_detailed = (
        fhs_loc_meta
        .query("most_detailed == 1")
        .loc[:, ['location_id', 'ihme_loc_id', 'local_id', 'location_name']]
        .rename(columns = {'location_id':'fhs_location_id', 'ihme_loc_id':'fhs_ihme_loc_id', 'local_id':'fhs_local_id', 'location_name':'fhs_location_name'})
    )
    
    lsae_loc_most_detailed['hierarchy'] = lsae_loc_most_detailed['path_to_top_parent'].str.split(',')
    exploded_lsae_loc = lsae_loc_most_detailed.explode('hierarchy')
    exploded_lsae_loc['hierarchy'] = exploded_lsae_loc['hierarchy'].astype(int)
    exploded_lsae_loc = exploded_lsae_loc.merge(
        lsae_loc_meta[['location_id', 'location_name']].rename(columns = {'location_id': 'hierarchy', 'location_name':'lsae_hierarchy_name'}), 
        on='hierarchy', 
        how='left',
    )
    lsae_fhs_mapping = exploded_lsae_loc.merge(
        fhs_loc_most_detailed, 
        left_on='hierarchy', 
        right_on='fhs_location_id', 
        how='right',
    )
    
    fhs_loc_meta.loc[fhs_loc_meta.location_id == SNNP_lsae_locid, 'location_id'] = SNNP_fhs_locid
    fhs_loc_most_detailed.loc[fhs_loc_most_detailed.fhs_location_id == SNNP_lsae_locid, 'fhs_location_id'] = SNNP_fhs_locid
    lsae_fhs_mapping.loc[lsae_fhs_mapping.fhs_location_id == SNNP_lsae_locid, 'fhs_location_id'] = SNNP_fhs_locid
    
    # Now, FHS, check that the locations match between the fhs shapefile, the most detailed loc metadata and the forecasted population data
    assert set(fhs_loc_most_detailed.fhs_location_id.unique()) == set(lsae_fhs_mapping.fhs_location_id.unique())
    
    lsae_fhs_mapping['worldpop_iso3'] = lsae_fhs_mapping.fhs_ihme_loc_id.str[0:3]
    # For worldpop, make sure all the locations we need are included in the worldpop data
    worldpop_locs = [name.parts[-1] for name in WORLDPOP_FILEPATH.glob('*/')]
    assert len(set(lsae_fhs_mapping.worldpop_iso3.unique()) - set(worldpop_locs)) == 0
    
    assert len(set(fhs_loc_most_detailed.fhs_location_id) - set(fhs_shapes.loc_id)) == 0

    lsae_fhs_mapping[['fhs_location_id', 'lsae_location_id', 'worldpop_iso3']].to_parquet(LSAE_FHS_MAPPING_PATH)


def load_fhs_lsae_mapping(fhs_loc_id: int = None):
    mapping_filters = [('fhs_location_id', '==', fhs_loc_id)] if fhs_loc_id else None
    fhs_lsae_mapping = pd.read_parquet(LSAE_FHS_MAPPING_PATH, filters=mapping_filters)

    fhs_filters = [("loc_id", "in", fhs_lsae_mapping.fhs_location_id.unique().tolist())]
    lsae_filters = [("loc_id", "in", fhs_lsae_mapping.lsae_location_id.unique().tolist())]    
        
    fhs_shapes = (
        gpd.read_parquet(FHS_SHAPE_PATH, filters=fhs_filters)
        .loc[:, ['loc_id', 'geometry']]
        .rename(columns={'geometry': 'fhs_shape'})
    )
    lsae_shapes = (
        gpd.read_parquet(LSAE_SHAPE_PATH, filters=lsae_filters)
        .loc[:, ['loc_id', 'geometry']]
        .rename(columns={'geometry': 'lsae_shape'})
    )    

    fhs_lsae_mapping = (
        fhs_lsae_mapping
        .merge(fhs_shapes, left_on='fhs_location_id', right_on='loc_id', how='inner')
        .merge(lsae_shapes, left_on='lsae_location_id', right_on='loc_id', how='inner')
        .drop(columns=['loc_id_x', 'loc_id_y'])
    )
    return gpd.GeoDataFrame(fhs_lsae_mapping, geometry='fhs_shape')