import rasterra as rt
import xarray as xr
import pandas as pd
import numpy as np

from rasterio.features import rasterize
from rra_climate_health.model_specification import ModelSpecification
from rra_climate_health.data import ClimateMalnutritionData

def xarray_to_raster(ds: xr.DataArray, nodata: float | int) -> rt.RasterArray:
    from affine import Affine

    """Convert an xarray DataArray to a RasterArray."""
    lat, lon = ds["latitude"].data, ds["longitude"].data

    dlat = (lat[1:] - lat[:-1]).mean()
    dlon = (lon[1:] - lon[:-1]).mean()

    transform = Affine(
        a=dlon,
        b=0.0,
        c=lon[0],
        d=0.0,
        e=-dlat,
        f=lat[-1],
    )
    raster = rt.RasterArray(
        data=ds.data[::-1],
        transform=transform,
        crs="EPSG:4326",
        no_data_value=nodata,
    )
    return raster


def get_intercept_raster(
    model_spec: ModelSpecification,
    coefs: pd.DataFrame,
    ranefs: pd.DataFrame,
    cm_data: ClimateMalnutritionData,
) -> rt.RasterArray:
    pred_spec = next(
            (x for x in model_spec.predictors if x.name == 'intercept'), None
        )
    if pred_spec is None:
        error_message = "No intercept predictor found in model specification"
        raise ValueError(error_message)
    raster_template = cm_data.load_raster_template()
    icept = coefs.loc["(Intercept)"]["Estimate"]
    if pred_spec.random_effect == "ihme_loc_id":
        fhs_shapes = cm_data.load_fhs_shapes()
        shapes = list(
            ranefs["X.Intercept."]
            .reset_index()
            .merge(fhs_shapes, left_on="index", right_on="ihme_lc_id", how="left")
            .loc[:, ["geometry", "X.Intercept."]]
            .itertuples(index=False, name=None)
        )
        icept_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept + icept_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
        )
    elif pred_spec.random_effect == "lbd_admin2_id":
        fhs_shapes = cm_data.load_lbd_admin2_shapes()
        shapes = list(
            ranefs["X.Intercept."]
            .reset_index()
            .merge(fhs_shapes, left_on="index", right_on="loc_id", how="left")
            .loc[:, ["geometry", "X.Intercept."]]
            .itertuples(index=False, name=None)
        )
        icept_arr = rasterize(
            shapes,
            out=np.zeros_like(raster_template),
            transform=raster_template.transform,
        )
        icept_raster = rt.RasterArray(
            icept + icept_arr,
            transform=raster_template.transform,
            crs=raster_template.crs,
            no_data_value=np.nan,
        )
    elif not pred_spec.random_effect:
        icept_raster = raster_template + icept
    else:
        msg = "Only location random intercepts are supported"
        raise NotImplementedError(msg)
    return icept_raster