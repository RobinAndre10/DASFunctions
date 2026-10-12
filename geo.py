"""Coordinate conversions (lat/lon <-> UTM, Norway/Svalbard zones) and map tiles."""
import numpy as np
import utm
from pyproj import CRS, Transformer
import cartopy.io.img_tiles as cimgt

__all__ = [
    "geocoord2utmcoord",
    "utmcorrd2geocoord",
    "computeDistanceFrom_Lat_Lon",
    "latlon_to_utm_svalbard",
    "latlon_to_utm_no",
    "utm_to_latlon",
    "get_basemap",
]


def geocoord2utmcoord(lat,
                      lon
                      ):
    """
    Simple script converting latitude longitude pair to utm pair.
    """
    easting, northing, zone_number, zone_letter = utm.from_latlon(np.asarray(lat), np.asarray(lon))

    return easting, northing, zone_number, zone_letter


def utmcorrd2geocoord(easting,
                      northing,
                      zone_number,
                      zone_letter
                      ):
    """
    Simple script converting utm pair to latitude longitude pair.
    """
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)

    return lat, lon


def computeDistanceFrom_Lat_Lon(lat,
                      lon
                      ):
    """
    Script computing the distance between point and accumulated distances based on Lat, Lon vectors.
    """
    easting, northing, zone_number, zone_letter = utm.from_latlon(np.asarray(lat), np.asarray(lon))
    distances = ( np.sqrt( np.diff(easting)**2 + np.diff(northing)**2 ) )
    cum_distances = np.cumsum( distances )
    return distances, cum_distances


def latlon_to_utm_svalbard(lat,
                           lon
                           ):
    """
    Converts latitude/longitude to UTM coordinates in the correct Svalbard zone.
    """

    # Determine correct UTM zone based on longitude
    if 0 <= lon < 9:
        zone = 31
    elif 9 <= lon < 21:
        zone = 33
    elif 21 <= lon < 33:
        zone = 35
    elif 33 <= lon < 42:
        zone = 37
    else:
        raise ValueError(f"Longitude {lon}° out of Svalbard UTM zone range (0–42°E).")

    # Define CRS for WGS84 and target UTM
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(32600 + zone)

    # Transform coordinates
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    easting, northing = transformer.transform(lon, lat)

    return easting, northing, zone


def _pick_utm_zone_norway_svalbard(lat: float,
                                   lon: float
                                   ) -> int:
    """
    Choose UTM zone for WGS84 in Norway + Svalbard region, applying UTM exceptions:
      - Svalbard (72–84N): 31/33/35/37 by lon
      - Norway band V exception (56–64N & 3–12E): force zone 32
    Falls back to standard UTM zone elsewhere.
    """
    # Normalize longitude to [-180, 180)
    lon = ((lon + 180) % 360) - 180

    # Svalbard special zones (lat band X, 72–84N)
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            return 31
        elif 9 <= lon < 21:
            return 33
        elif 21 <= lon < 33:
            return 35
        elif 33 <= lon < 42:
            return 37
        else:
            # Outside the Svalbard special longitudes; default to standard UTM
            pass

    # Norway band V exception (56–64N, 3–12E) -> zone 32
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32

    # Standard UTM zone
    zone = int(np.floor((lon + 180) / 6) + 1)
    # Clamp to [1, 60] just in case
    zone = max(1, min(60, zone))
    return zone


def latlon_to_utm_no(lat: float,
                     lon: float
                     ):
    """
    Convert latitude/longitude (WGS84) to UTM easting/northing/zone
    for mainland Norway + Svalbard, applying regional UTM exceptions.
    Returns (easting, northing, zone).
    """
    zone = _pick_utm_zone_norway_svalbard(lat, lon)
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(32600 + zone)  # Northern hemisphere UTM on WGS84
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing, zone


def utm_to_latlon(easting: float,
                  northing: float,
                  zone: int
                  ):
    """
    Convert UTM (easting, northing, zone) to latitude/longitude (WGS84).
    Assumes northern hemisphere (appropriate for Norway + Svalbard).
    Returns (lat, lon).
    """
    if not (1 <= zone <= 60):
        raise ValueError("UTM zone must be in 1..60.")
    # All of Norway and Svalbard are in the northern hemisphere -> EPSG 326xx
    crs_utm = CRS.from_epsg(32600 + zone)
    crs_wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)
    lon, lat = transformer.transform(easting, northing)

    return lat, lon


def get_basemap(name="osm"):
    """
    Returns a Cartopy tile source (img_tiles) for a given basemap name.
    Works across Cartopy versions by trying providers in a safe order.

    Options: 'osm', 'stamen_terrain', 'stamen_toner', 'stamen_watercolor',
             'esri_world_imagery', 'esri_world_topo'
    """
    name = name.lower()

    # --- Always available in older Cartopy ---
    if name in ("osm", "openstreetmap"):
        return cimgt.OSM()

    # --- Stamen (availability depends on Cartopy version) ---
    # Some Cartopy versions have Stamen(), some have StamenTerrain etc.
    if name in ("stamen_terrain", "terrain"):
        if hasattr(cimgt, "Stamen"):
            return cimgt.Stamen("terrain-background")
        elif hasattr(cimgt, "StamenTerrain"):
            return cimgt.StamenTerrain()
        else:
            raise ValueError("Stamen terrain tiles not available in this Cartopy version.")

    if name in ("stamen_toner", "toner"):
        if hasattr(cimgt, "Stamen"):
            return cimgt.Stamen("toner")
        else:
            raise ValueError("Stamen toner tiles not available in this Cartopy version.")

    if name in ("stamen_watercolor", "watercolor"):
        if hasattr(cimgt, "Stamen"):
            return cimgt.Stamen("watercolor")
        else:
            raise ValueError("Stamen watercolor tiles not available in this Cartopy version.")

    # --- ESRI (not always included; some versions have GoogleWTS / QuadtreeTiles etc.) ---
    # If your Cartopy has 'GoogleWTS', we can use ESRI via its URL template.
    if name in ("esri_world_imagery", "satellite", "imagery"):
        if hasattr(cimgt, "GoogleWTS"):
            class ESRIWorldImagery(cimgt.GoogleWTS):
                def _image_url(self, tile):
                    x, y, z = tile
                    return (f"https://services.arcgisonline.com/ArcGIS/rest/services/"
                            f"World_Imagery/MapServer/tile/{z}/{y}/{x}")
            return ESRIWorldImagery()
        else:
            raise ValueError("ESRI tiles require cimgt.GoogleWTS in your Cartopy version.")

    if name in ("esri_world_topo", "topo"):
        if hasattr(cimgt, "GoogleWTS"):
            class ESRIWorldTopo(cimgt.GoogleWTS):
                def _image_url(self, tile):
                    x, y, z = tile
                    return (f"https://services.arcgisonline.com/ArcGIS/rest/services/"
                            f"World_Topo_Map/MapServer/tile/{z}/{y}/{x}")
            return ESRIWorldTopo()
        else:
            raise ValueError("ESRI tiles require cimgt.GoogleWTS in your Cartopy version.")

    raise ValueError(f"Unknown basemap '{name}'.")
