"""Map view component for spatial visualizations."""
import plotly.express as px
import pandas as pd
from typing import Optional


def create_station_map(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    color_col: str = "utilization",
    size_col: Optional[str] = None,
    center_lat: float = 22.62,
    center_lon: float = 114.07,
    zoom: int = 10,
    colorscale: str = "YlOrRd",
    height: int = 600,
    title: str = "",
):
    """Create an interactive mapbox scatter plot of stations."""
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        size=size_col,
        color_continuous_scale=colorscale,
        mapbox_style="carto-darkmatter",
        zoom=zoom,
        center={"lat": center_lat, "lon": center_lon},
        height=height,
        title=title,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig
