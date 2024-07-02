

def run_ldi_prep_main(
    year: int,
) -> None:
    """Run LDI data preparation."""
    data = LdiData()
    data.download_ldi_data(year)
    data.process_ldi_data(year)