



def equal_intervals_pandas_series(series, nbins=10):
    import pandas as pd
    
    max = series.max()
    min = series.min()

    bin_size = (max - min) / nbins

    for bin_id in range(nbins):
        bin_bounds = [min + bin_id * bin_size,
                      min + (bin_id + 1) * bin_size]
        series = series.apply(lambda x: bin_bounds[0] if x >= bin_bounds[0] and x < bin_bounds[1] else x)

    return series

