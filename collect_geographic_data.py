"""
collect_geographic_data.py
Geographic service data collection for the Milan Real Estate Value-for-Money project.

Queries OpenStreetMap via the Overpass API for six service types across Milan,
spatially joins them to neighbourhood polygons, then aggregates counts and
computes a normalised service-accessibility score for both the 23 crime-index
macro-zones and the 32 Immobiliare.it districts.

Outputs (saved to data/):
    services_raw.csv                      all POIs with coordinates and type
    services_zoned.csv                    POIs with OSM neighbourhood attributed
    service_scores_crime_zones.csv        counts + scores for 23 crime-index zones
    service_scores_immobiliare_zones.csv  counts + scores for 32 Immobiliare.it zones

Requirements:
    pip install requests geopandas osmnx pandas shapely
"""

import sys
import time
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Force UTF-8 output so accented characters (à, è, etc.) print correctly on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Approximate bounding box for the Municipality of Milan (south, west, north, east).
MILAN_BBOX = "45.38,9.04,45.54,9.28"

# Weights for the composite service-accessibility score.
# Assumption: metro connectivity is the strongest driver of accessibility;
# hospitals and schools are high-impact but less spatially variable than
# transport; supermarkets and parks carry lower weight.
SERVICE_WEIGHTS = {
    "metro_stations": 5,
    "tram_stops":     3,
    "hospitals":      4,
    "supermarkets":   2,
    "schools":        3,
    "parks":          2,
}

# =============================================================================
# REFERENCE ZONE LISTS
# Used to guarantee every zone appears in the output (even with zero counts).
# =============================================================================

CRIME_ZONES_23 = [
    "Brera", "Corvetto", "San Siro", "Duomo", "Porta Venezia",
    "Centrale", "Loreto – Via Padova", "Navigli", "Giambellino-Lorenteggio",
    "Porta Garibaldi", "Rogoredo", "Lambrate", "Gratosoglio", "Quarto Oggiaro",
    "Corso Como – Isola", "Baggio", "Porta Romana", "Bruzzano", "Bicocca",
    "Città Studi", "Gorla", "Sempione", "Porta Ticinese",
]

IMMOBILIARE_ZONES_32 = [
    "Centro",
    "Garibaldi, Moscova, Porta Nuova",
    "Arco della Pace, Arena, Pagano",
    "Quadronno, Palestro, Guastalla",
    "Genova, Ticinese",
    "Porta Venezia, Indipendenza",
    "Porta Romana, Cadore, Montenero",
    "Solari, Washington",
    "Fiera, Sempione, City Life, Portello",
    "Centrale, Repubblica",
    "Navigli",
    "Corsico, Sarpi, Isola",
    "Città Studi, Susa",
    "Napoli, Soderini",
    "Maggiolina, Istria",
    "Porta Vittoria, Lodi",
    "Bande Nere, Inganni",
    "Ripamonti, Vigentino",
    "Pasteur, Rovereto",
    "Precotto, Turro",
    "Famagosta, Barona",
    "Udine, Lambrate",
    "Abbiategrasso, Chiesa Rossa",
    "Corvetto, Rogoredo",
    "Uptown, Cascina Merlata, Viale Certosa",
    "San Siro, Trenno",
    "Bicocca, Niguarda",
    "Affori, Bovisa",
    "Cimiano, Crescenzago, Adriano",
    "Forlanini",
    "Bisceglie, Baggio, Olmi",
    "Ponte Lambro, Santa Giulia",
]

# =============================================================================
# CROSSWALK TABLES
# Maps OSM neighbourhood name -> target zone name.
#
# IMPORTANT: Run the script once and read the printed list of OSM names
# (Section 2 output). If any neighbourhood name is missing here, add it.
# OSM names that are absent from the crosswalk will be silently dropped.
# =============================================================================

OSM_TO_CRIME_ZONE = {
    # Keys are the exact OSM neighbourhood names returned by osmnx (Section 2 output).
    # --- Duomo ---
    "Duomo":                            "Duomo",
    "Guastalla":                        "Duomo",
    # --- Brera ---
    "Brera":                            "Brera",
    # --- Porta Garibaldi ---
    "GARIBALDI REPUBBLICA":             "Porta Garibaldi",
    # --- Corso Como - Isola ---
    "Isola":                            "Corso Como – Isola",
    "Sarpi":                            "Corso Como – Isola",
    "Farini":                           "Corso Como – Isola",
    # --- Navigli ---
    "Navigli":                          "Navigli",
    "Parco dei Navigli":                "Navigli",
    "San Cristoforo":                   "Navigli",
    "Ronchetto Sul Naviglio":           "Navigli",
    "Ronchetto sul Naviglio":           "Navigli",
    "Ronchetto delle Rane":             "Navigli",
    "Barona":                           "Navigli",
    "Washington":                       "Navigli",
    # --- Porta Ticinese ---
    "Ticinese":                         "Porta Ticinese",
    "Tortona":                          "Porta Ticinese",
    # --- Porta Venezia ---
    "Buenos Aires - Venezia":           "Porta Venezia",
    "Giardini Porta Venezia":           "Porta Venezia",
    "XXII Marzo":                       "Porta Venezia",
    "Corsica":                          "Porta Venezia",
    # --- Centrale ---
    "Centrale":                         "Centrale",
    "Nolo":                             "Centrale",
    # --- Loreto - Via Padova ---
    "Loreto":                           "Loreto – Via Padova",
    "Padova":                           "Loreto – Via Padova",
    "Viale Monza":                      "Loreto – Via Padova",
    "Greco":                            "Loreto – Via Padova",
    # --- Sempione ---
    "Sempione":                         "Sempione",
    "Parco Sempione":                   "Sempione",
    "Portello":                         "Sempione",
    "Qt 8":                             "Sempione",
    "Bullona":                          "Sempione",
    "De Angeli - Monte Rosa":           "Sempione",
    "Pagano":                           "Sempione",
    "Lampugnano":                       "Sempione",
    "Tre Torri":                        "Sempione",
    "Magenta - San Vittore":            "Sempione",
    # --- Citta Studi ---
    "Città Studi":                 "Città Studi",
    # --- Lambrate ---
    "Lambrate":                         "Lambrate",
    "Parco Lambro - Cimiano":           "Lambrate",
    "Parco Forlanini - Ortica":         "Lambrate",
    # --- Gorla (Gorla/Turro/Precotto have no OSM polygon; partial coverage via adjacent zones) ---
    "Adriano":                          "Gorla",
    # --- Bicocca ---
    "Bicocca":                          "Bicocca",
    "Niguarda - Cà Granda":        "Bicocca",
    "Maciachini - Maggiolina":          "Bicocca",
    "Parco Nord":                       "Bicocca",
    "Parco Bosco In Città":        "Bicocca",
    # --- Bruzzano ---
    "Bruzzano":                         "Bruzzano",
    "Affori":                           "Bruzzano",
    "Bovisa":                           "Bruzzano",
    "Dergano":                          "Bruzzano",
    "Brusuglio":                        "Bruzzano",
    "Stephenson":                       "Bruzzano",
    # --- Porta Romana ---
    "Porta Romana":                     "Porta Romana",
    "Scalo Romana":                     "Porta Romana",
    "Vigentina":                        "Porta Romana",
    # --- San Siro ---
    "San Siro":                         "San Siro",
    "Trenno":                           "San Siro",
    # --- Giambellino-Lorenteggio ---
    "Giambellino":                      "Giambellino-Lorenteggio",
    "Lorenteggio":                      "Giambellino-Lorenteggio",
    "Bande Nere":                       "Giambellino-Lorenteggio",
    "Sacco":                            "Giambellino-Lorenteggio",
    "Forze Armate":                     "Giambellino-Lorenteggio",
    "Quartiere Aldini I":               "Giambellino-Lorenteggio",
    "Quartiere Aldini II":              "Giambellino-Lorenteggio",
    # --- Corvetto ---
    "Lodi - Corvetto":                  "Corvetto",
    "Umbria - Molise":                  "Corvetto",
    "Chiaravalle":                      "Corvetto",
    # --- Rogoredo ---
    "Rogoredo":                         "Rogoredo",
    "Mecenate":                         "Rogoredo",
    "Ortomercato":                      "Rogoredo",
    "Quartiere Forlanini":              "Rogoredo",
    "Parco Monlué - Ponte Lambro": "Rogoredo",
    "East Garden (ex De Nora)":         "Rogoredo",
    # --- Gratosoglio ---
    "Gratosoglio":                      "Gratosoglio",
    "Gratosoglio - Ticinello":          "Gratosoglio",
    "Chiesa Rossa":                     "Gratosoglio",
    "Moncucco":                         "Gratosoglio",
    "Stadera":                          "Gratosoglio",
    "Tibaldi":                          "Gratosoglio",
    "Morivione":                        "Gratosoglio",
    "Ex Om - Morivione":                "Gratosoglio",
    "Ripamonti":                        "Gratosoglio",
    # --- Quarto Oggiaro ---
    "Quarto Oggiaro":                   "Quarto Oggiaro",
    "Comasina":                         "Quarto Oggiaro",
    "Quartiere IACP Quarto Oggiaro":    "Quarto Oggiaro",
    "Villapizzone":                     "Quarto Oggiaro",
    "Ghisolfa":                         "Quarto Oggiaro",
    "Roserio":                          "Quarto Oggiaro",
    "Maggiore - Musocco":               "Quarto Oggiaro",
    "Quartiere Vialba I":               "Quarto Oggiaro",
    "Torri di Via Lessona":             "Quarto Oggiaro",
    # --- Baggio ---
    "Baggio":                           "Baggio",
    "Selinunte":                        "Baggio",
    "Gallaratese":                      "Baggio",
    "Quartiere Gallaratese":            "Baggio",
    "Cantalupa":                        "Baggio",
    "Quinto Romano":                    "Baggio",
    "Quarto Cagnino":                   "Baggio",
    "Muggiano":                         "Baggio",
    "Figino":                           "Baggio",
}

OSM_TO_IMMOBILIARE_ZONE = {
    # Keys are the exact OSM neighbourhood names returned by osmnx (Section 2 output).
    # --- Centro ---
    "Duomo":                            "Centro",
    "Guastalla":                        "Quadronno, Palestro, Guastalla",
    # --- Garibaldi, Moscova, Porta Nuova ---
    "Brera":                            "Garibaldi, Moscova, Porta Nuova",
    "GARIBALDI REPUBBLICA":             "Garibaldi, Moscova, Porta Nuova",
    # --- Arco della Pace, Arena, Pagano ---
    "Parco Sempione":                   "Arco della Pace, Arena, Pagano",
    "Bullona":                          "Arco della Pace, Arena, Pagano",
    "De Angeli - Monte Rosa":           "Arco della Pace, Arena, Pagano",
    "Pagano":                           "Arco della Pace, Arena, Pagano",
    "Magenta - San Vittore":            "Arco della Pace, Arena, Pagano",
    # --- Quadronno, Palestro, Guastalla (already mapped above) ---
    # --- Genova, Ticinese ---
    "Ticinese":                         "Genova, Ticinese",
    "Tortona":                          "Genova, Ticinese",
    "San Cristoforo":                   "Genova, Ticinese",
    "Ronchetto Sul Naviglio":           "Genova, Ticinese",
    "Ronchetto sul Naviglio":           "Genova, Ticinese",
    "Ronchetto delle Rane":             "Genova, Ticinese",
    # --- Porta Venezia, Indipendenza ---
    "Buenos Aires - Venezia":           "Porta Venezia, Indipendenza",
    "Giardini Porta Venezia":           "Porta Venezia, Indipendenza",
    "XXII Marzo":                       "Porta Venezia, Indipendenza",
    "Corsica":                          "Porta Venezia, Indipendenza",
    # --- Porta Romana, Cadore, Montenero ---
    "Porta Romana":                     "Porta Romana, Cadore, Montenero",
    "Scalo Romana":                     "Porta Romana, Cadore, Montenero",
    "Vigentina":                        "Porta Romana, Cadore, Montenero",
    # --- Solari, Washington ---
    "Washington":                       "Solari, Washington",
    "Barona":                           "Solari, Washington",
    # --- Fiera, Sempione, City Life, Portello ---
    "Sempione":                         "Fiera, Sempione, City Life, Portello",
    "Portello":                         "Fiera, Sempione, City Life, Portello",
    "Qt 8":                             "Fiera, Sempione, City Life, Portello",
    "Lampugnano":                       "Fiera, Sempione, City Life, Portello",
    "Tre Torri":                        "Fiera, Sempione, City Life, Portello",
    # --- Centrale, Repubblica ---
    "Centrale":                         "Centrale, Repubblica",
    "Nolo":                             "Centrale, Repubblica",
    # --- Navigli ---
    "Navigli":                          "Navigli",
    "Parco dei Navigli":                "Navigli",
    # --- Corsico, Sarpi, Isola ---
    "Isola":                            "Corsico, Sarpi, Isola",
    "Sarpi":                            "Corsico, Sarpi, Isola",
    "Farini":                           "Corsico, Sarpi, Isola",
    # --- Citta Studi, Susa ---
    "Città Studi":                 "Città Studi, Susa",
    # --- Napoli, Soderini (no direct OSM polygon match found) ---
    # --- Maggiolina, Istria ---
    "Maciachini - Maggiolina":          "Maggiolina, Istria",
    "Greco":                            "Maggiolina, Istria",
    # --- Porta Vittoria, Lodi ---
    "XXII Marzo":                       "Porta Vittoria, Lodi",
    # --- Bande Nere, Inganni ---
    "Bande Nere":                       "Bande Nere, Inganni",
    "Giambellino":                      "Bande Nere, Inganni",
    "Lorenteggio":                      "Bande Nere, Inganni",
    "Sacco":                            "Bande Nere, Inganni",
    "Forze Armate":                     "Bande Nere, Inganni",
    "Quartiere Aldini I":               "Bande Nere, Inganni",
    "Quartiere Aldini II":              "Bande Nere, Inganni",
    # --- Ripamonti, Vigentino ---
    "Ripamonti":                        "Ripamonti, Vigentino",
    "Morivione":                        "Ripamonti, Vigentino",
    "Ex Om - Morivione":                "Ripamonti, Vigentino",
    # --- Pasteur, Rovereto ---
    "Loreto":                           "Pasteur, Rovereto",
    "Viale Monza":                      "Pasteur, Rovereto",
    # --- Precotto, Turro ---
    "Adriano":                          "Precotto, Turro",
    # --- Famagosta, Barona ---
    "Famagosta":                        "Famagosta, Barona",
    # --- Udine, Lambrate ---
    "Lambrate":                         "Udine, Lambrate",
    "Parco Forlanini - Ortica":         "Udine, Lambrate",
    "Parco Lambro - Cimiano":           "Udine, Lambrate",
    # --- Abbiategrasso, Chiesa Rossa ---
    "Gratosoglio":                      "Abbiategrasso, Chiesa Rossa",
    "Gratosoglio - Ticinello":          "Abbiategrasso, Chiesa Rossa",
    "Chiesa Rossa":                     "Abbiategrasso, Chiesa Rossa",
    "Moncucco":                         "Abbiategrasso, Chiesa Rossa",
    "Stadera":                          "Abbiategrasso, Chiesa Rossa",
    "Tibaldi":                          "Abbiategrasso, Chiesa Rossa",
    # --- Corvetto, Rogoredo ---
    "Lodi - Corvetto":                  "Corvetto, Rogoredo",
    "Umbria - Molise":                  "Corvetto, Rogoredo",
    "Chiaravalle":                      "Corvetto, Rogoredo",
    "Rogoredo":                         "Corvetto, Rogoredo",
    # --- Uptown, Cascina Merlata, Viale Certosa ---
    "Quarto Oggiaro":                   "Uptown, Cascina Merlata, Viale Certosa",
    "Comasina":                         "Uptown, Cascina Merlata, Viale Certosa",
    "Quartiere IACP Quarto Oggiaro":    "Uptown, Cascina Merlata, Viale Certosa",
    "Villapizzone":                     "Uptown, Cascina Merlata, Viale Certosa",
    "Ghisolfa":                         "Uptown, Cascina Merlata, Viale Certosa",
    "Roserio":                          "Uptown, Cascina Merlata, Viale Certosa",
    "Maggiore - Musocco":               "Uptown, Cascina Merlata, Viale Certosa",
    "Quartiere Vialba I":               "Uptown, Cascina Merlata, Viale Certosa",
    "Torri di Via Lessona":             "Uptown, Cascina Merlata, Viale Certosa",
    # --- San Siro, Trenno ---
    "San Siro":                         "San Siro, Trenno",
    "Trenno":                           "San Siro, Trenno",
    # --- Bicocca, Niguarda ---
    "Bicocca":                          "Bicocca, Niguarda",
    "Niguarda - Cà Granda":        "Bicocca, Niguarda",
    "Parco Nord":                       "Bicocca, Niguarda",
    "Parco Bosco In Città":        "Bicocca, Niguarda",
    # --- Affori, Bovisa ---
    "Affori":                           "Affori, Bovisa",
    "Bovisa":                           "Affori, Bovisa",
    "Dergano":                          "Affori, Bovisa",
    "Brusuglio":                        "Affori, Bovisa",
    "Stephenson":                       "Affori, Bovisa",
    # --- Cimiano, Crescenzago, Adriano (no distinct OSM polygon; Adriano used above) ---
    # --- Forlanini ---
    "Mecenate":                         "Forlanini",
    "Ortomercato":                      "Forlanini",
    "Quartiere Forlanini":              "Forlanini",
    "East Garden (ex De Nora)":         "Forlanini",
    # --- Bisceglie, Baggio, Olmi ---
    "Baggio":                           "Bisceglie, Baggio, Olmi",
    "Selinunte":                        "Bisceglie, Baggio, Olmi",
    "Gallaratese":                      "Bisceglie, Baggio, Olmi",
    "Quartiere Gallaratese":            "Bisceglie, Baggio, Olmi",
    "Cantalupa":                        "Bisceglie, Baggio, Olmi",
    "Quinto Romano":                    "Bisceglie, Baggio, Olmi",
    "Quarto Cagnino":                   "Bisceglie, Baggio, Olmi",
    "Muggiano":                         "Bisceglie, Baggio, Olmi",
    "Figino":                           "Bisceglie, Baggio, Olmi",
    # --- Ponte Lambro, Santa Giulia ---
    "Parco Monlué - Ponte Lambro": "Ponte Lambro, Santa Giulia",
}

# =============================================================================
# SECTION 1 — FETCH SERVICE POINTS FROM OVERPASS API
# =============================================================================

# Maps (tag_key, tag_value) pairs found in OSM elements to our service type labels.
_TAG_TO_SERVICE = {
    ("station",  "subway"):     "metro_stations",
    ("railway",  "tram_stop"):  "tram_stops",
    ("amenity",  "hospital"):   "hospitals",
    ("shop",     "supermarket"):"supermarkets",
    ("amenity",  "school"):     "schools",
    ("leisure",  "park"):       "parks",
}


_HEADERS = {
    # Identify the client to the public Overpass server — avoids 406 rejections.
    "User-Agent": "MilanRealEstateProject/1.0 (university research; contact: bissio04.fs@gmail.com)",
}


def _run_overpass(overpass_ql: str, retries: int = 3) -> list:
    """
    POST a raw Overpass QL body to the API and return the elements list.
    Retries up to `retries` times with exponential back-off on server errors.
    Relations are intentionally excluded from all queries: complex multipolygon
    relations combined with 'out center' trigger 406 errors on the public server.
    """
    full_query = f"[out:json][timeout:90];\n({overpass_ql});\nout center;"
    for attempt in range(retries):
        time.sleep(2 + attempt * 5)
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": full_query},
                headers=_HEADERS,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["elements"]
        except requests.HTTPError as exc:
            if attempt < retries - 1:
                print(f"  Overpass error ({exc}) — retrying (attempt {attempt + 2}/{retries})...")
            else:
                raise
    return []


def fetch_all_services(bbox: str) -> pd.DataFrame:
    """
    Query all six service types inside the Milan bounding box in a single
    Overpass request and return a tidy DataFrame.
    Relations are excluded: nodes and ways cover the vast majority of OSM
    features in Milan; relations add complexity without meaningful coverage gain.
    """
    ql = f"""
      node["station"="subway"]({bbox});
      node["railway"="tram_stop"]({bbox});
      node["amenity"="hospital"]({bbox});
      way["amenity"="hospital"]({bbox});
      node["shop"="supermarket"]({bbox});
      node["amenity"="school"]({bbox});
      way["amenity"="school"]({bbox});
      way["leisure"="park"]({bbox});
    """
    print("Section 1: querying Overpass API for all service types...")
    elements = _run_overpass(ql)

    records = []
    for el in elements:
        tags = el.get("tags", {})
        # ways and relations carry geometry under a 'center' key when using 'out center'
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        service_type = next(
            (stype for (k, v), stype in _TAG_TO_SERVICE.items() if tags.get(k) == v),
            None,
        )
        if service_type is None:
            continue
        records.append({
            "osm_id":       el["id"],
            "osm_type":     el["type"],
            "service_type": service_type,
            "name":         tags.get("name", ""),
            "lat":          lat,
            "lon":          lon,
        })

    df = pd.DataFrame(records)
    print(f"  -> {len(df)} features collected.")
    print(f"  -> Breakdown: {df['service_type'].value_counts().to_dict()}\n")
    return df


# =============================================================================
# SECTION 2 — FETCH ZONE POLYGONS FROM OSM
# osmnx wraps the Overpass API and handles all geometry parsing.
# The returned names are printed so you can check and extend the crosswalk dicts.
# =============================================================================

def fetch_zone_polygons() -> gpd.GeoDataFrame:
    """
    Retrieve neighbourhood and quarter polygons for Milan from OSM using osmnx.
    Prints all returned OSM names — compare with crosswalk dicts and add any missing entries.
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError(
            "osmnx is required for zone polygon retrieval.\n"
            "Install with:  pip install osmnx"
        )

    print("Section 2: fetching zone polygons from OSM (via osmnx / Overpass)...")
    gdf = ox.features_from_place(
        "Milan, Italy",
        tags={"place": ["quarter", "neighbourhood", "suburb"]},
    )

    # Keep only area geometries (drop point nodes returned for some entries).
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf[["geometry", "name"]].dropna(subset=["name"]).reset_index(drop=True)
    gdf = gdf.to_crs("EPSG:4326")

    osm_names = sorted(gdf["name"].dropna().unique().tolist())
    print(f"  -> {len(gdf)} zone polygons found.")
    print(f"  -> OSM names returned:\n    {osm_names}")
    print(
        "\n  Check the crosswalk dicts (OSM_TO_CRIME_ZONE / OSM_TO_IMMOBILIARE_ZONE) "
        "and add any name above that is missing.\n"
    )
    return gdf


# =============================================================================
# SECTION 3 — SPATIAL JOIN: assign each POI to its OSM neighbourhood polygon
# =============================================================================

def assign_zones(services_df: pd.DataFrame, zones_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Point-in-polygon join: adds an 'osm_zone' column to the services DataFrame.
    Points that fall outside all zone polygons are dropped with a warning.
    """
    print("Section 3: running spatial join (point-in-polygon)...")
    services_gdf = gpd.GeoDataFrame(
        services_df,
        geometry=gpd.points_from_xy(services_df["lon"], services_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        services_gdf,
        zones_gdf[["geometry", "name"]].rename(columns={"name": "osm_zone"}),
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["geometry", "index_right"])

    n_unmatched = joined["osm_zone"].isna().sum()
    if n_unmatched:
        print(
            f"  Warning: {n_unmatched} POIs fell outside all zone polygons and will be dropped. "
            "This is expected for points near zone boundaries or in unmapped areas."
        )
    result = joined.dropna(subset=["osm_zone"]).reset_index(drop=True)
    print(f"  -> {len(result)} POIs successfully attributed to a zone.\n")
    return result


# =============================================================================
# SECTION 4 — AGGREGATE, CROSSWALK, AND SCORE
# =============================================================================

def count_and_score(
    services_zoned: pd.DataFrame,
    crosswalk: dict,
    zone_col: str,
    all_zones: list,
) -> pd.DataFrame:
    """
    Apply a crosswalk dict (OSM name -> target zone), count POIs per zone,
    compute a weighted raw score, and min-max normalise it to [0, 1].

    Zones in 'all_zones' that receive no mapped POIs are filled with zeros
    so every zone appears in the output.
    """
    df = services_zoned.copy()
    df[zone_col] = df["osm_zone"].map(crosswalk)

    n_unmapped = df[zone_col].isna().sum()
    if n_unmapped:
        unmapped_names = df.loc[df[zone_col].isna(), "osm_zone"].unique().tolist()
        print(
            f"  Warning [{zone_col}]: {n_unmapped} POIs in OSM zones not found in crosswalk "
            f"and will be dropped.\n  Unmapped OSM names: {unmapped_names}"
        )

    df = df.dropna(subset=[zone_col])

    # Count by zone × service_type, pivot to wide format
    counts = (
        df.groupby([zone_col, "service_type"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all service type columns exist (some may be entirely absent)
    for col in SERVICE_WEIGHTS:
        if col not in counts.columns:
            counts[col] = 0

    # Fill zones that received zero POIs after the crosswalk
    counts = counts.reindex(all_zones, fill_value=0)
    counts.index.name = zone_col

    # Weighted raw score
    counts["service_score_raw"] = sum(
        counts[col] * w for col, w in SERVICE_WEIGHTS.items()
    )

    # Min-max normalisation to [0, 1]
    mn, mx = counts["service_score_raw"].min(), counts["service_score_raw"].max()
    counts["service_score"] = (
        (counts["service_score_raw"] - mn) / (mx - mn) if mx > mn else 0.0
    )

    counts = counts.reset_index()
    print(f"\n  [{zone_col}] Scored {len(counts)} zones.")
    print(
        counts[[zone_col, "service_score_raw", "service_score"]]
        .sort_values("service_score", ascending=False)
        .to_string(index=False)
    )
    return counts


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # 1 — Service points
    services_raw = fetch_all_services(MILAN_BBOX)

    # 2 — Zone polygons (prints OSM names; update crosswalk dicts if needed)
    zones_gdf = fetch_zone_polygons()

    # 3 — Spatial join
    services_zoned = assign_zones(services_raw, zones_gdf)

    # 4a — 23 crime-index zones
    print("\nSection 4a: aggregating to 23 crime-index zones...")
    crime_scores = count_and_score(
        services_zoned, OSM_TO_CRIME_ZONE, "crime_zone", CRIME_ZONES_23
    )

    # 4b — 32 Immobiliare.it zones
    print("\nSection 4b: aggregating to 32 Immobiliare.it zones...")
    immobiliare_scores = count_and_score(
        services_zoned, OSM_TO_IMMOBILIARE_ZONE, "immobiliare_zone", IMMOBILIARE_ZONES_32
    )

    # 5 — Save (utf-8-sig adds a BOM so Excel on Windows opens the files correctly)
    services_raw.to_csv(DATA_DIR / "services_raw.csv", index=False, encoding="utf-8-sig")
    services_zoned.to_csv(DATA_DIR / "services_zoned.csv", index=False, encoding="utf-8-sig")
    crime_scores.to_csv(DATA_DIR / "service_scores_crime_zones.csv", index=False, encoding="utf-8-sig")
    immobiliare_scores.to_csv(DATA_DIR / "service_scores_immobiliare_zones.csv", index=False, encoding="utf-8-sig")

    print("\n--- Done. Files saved to data/ ---")
    print(f"  services_raw.csv                      {len(services_raw)} POIs")
    print(f"  services_zoned.csv                    {len(services_zoned)} POIs with zone")
    print(f"  service_scores_crime_zones.csv         {len(crime_scores)} zones")
    print(f"  service_scores_immobiliare_zones.csv   {len(immobiliare_scores)} zones")
