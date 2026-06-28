from pathlib import Path
import json
import folium
import rasterio
from rasterio.warp import transform_bounds, transform_geom
from PIL import Image

# project root (three levels up from src/.../pipelines)
BASE = Path(__file__).resolve().parents[3]
DATA = BASE / 'data'
PROCESSED = DATA / 'processed'
AOI = BASE / 'aoi.geojson'
GEOJSON = PROCESSED / 'change_features.geojson'
OUT_HTML = PROCESSED / 'part4_simple_map.html'
RGB_BEFORE = PROCESSED / 'sentinel2_20230812_stack_rgb_preview.tif'
RGB_BEFORE_PNG = PROCESSED / 'sentinel2_20230812_stack_rgb_preview.png'

# Get center from features or AOI
def get_center_from_features(gj):
    xs = []
    ys = []
    for f in gj.get('features', []):
        geom = f.get('geometry')
        if not geom: continue
        if geom['type'] == 'Polygon':
            for ring in geom['coordinates']:
                for x,y in ring:
                    xs.append(x); ys.append(y)
        elif geom['type'] == 'MultiPolygon':
            for poly in geom['coordinates']:
                for ring in poly:
                    for x,y in ring:
                        xs.append(x); ys.append(y)
    if xs and ys:
        return [(min(ys)+max(ys))/2, (min(xs)+max(xs))/2]
    return [0,0]

def run_part4_visualize():
    # load features
    if not GEOJSON.exists():
        raise SystemExit('Missing change_features.geojson: run Part 3 first')
    with open(GEOJSON, 'r', encoding='utf-8') as fh:
        feats = json.load(fh)

    # load AOI if present
    aoi_geom = None
    if AOI.exists():
        with open(AOI, 'r', encoding='utf-8') as fh:
            aoi = json.load(fh)
            if aoi.get('features'):
                aoi_geom = aoi['features'][0]['geometry']



    center = get_center_from_features(feats)
    m = folium.Map(location=center, zoom_start=12)

    # add raster overlay (ensure PNG exists)
    if RGB_BEFORE.exists():
        try:
            if not RGB_BEFORE_PNG.exists():
                with rasterio.open(RGB_BEFORE) as src:
                    arr = src.read([1,2,3]).astype('uint8').transpose((1,2,0))
                    Image.fromarray(arr).save(RGB_BEFORE_PNG)
                    rb = src.bounds; rcrs = src.crs
            else:
                with rasterio.open(RGB_BEFORE) as src:
                    rb = src.bounds; rcrs = src.crs
            if rcrs is not None and not getattr(rcrs, 'is_geographic', False):
                lon_min, lat_min, lon_max, lat_max = transform_bounds(rcrs, 'EPSG:4326', rb.left, rb.bottom, rb.right, rb.top)
            else:
                lon_min, lat_min, lon_max, lat_max = rb.left, rb.bottom, rb.right, rb.top
            img_bounds = [[lat_min, lon_min],[lat_max, lon_max]]
            folium.raster_layers.ImageOverlay(str(RGB_BEFORE_PNG), bounds=img_bounds, opacity=1.0).add_to(m)
        except Exception as e:
            print('Could not add image overlay:', e)

    # if AOI exists, add as overlay (reproject if needed)
    if aoi_geom is not None:
        try:
            folium.GeoJson({'type':'Feature','geometry':aoi_geom}, name='AOI', style_function=lambda x: {'color':'blue','weight':2,'fill':False}).add_to(m)
        except Exception:
            pass

    # reproject features if needed to 4326 using raster CRS if available
    stack_path = PROCESSED / 'sentinel2_20230812_stack.tif'
    if stack_path.exists():
        try:
            with rasterio.open(stack_path) as src:
                src_crs = src.crs
            if src_crs is not None and not getattr(src_crs, 'is_geographic', False):
                new_feats = []
                for feat in feats.get('features', []):
                    try:
                        new_geom = transform_geom(src_crs, 'EPSG:4326', feat['geometry'])
                        nf = feat.copy(); nf['geometry'] = new_geom; new_feats.append(nf)
                    except Exception:
                        new_feats.append(feat)
                feats['features'] = new_feats
        except Exception:
            pass

    # add polygons layer
    folium.GeoJson(feats, name='Change polygons', tooltip=folium.GeoJsonTooltip(fields=['area_m2','pixel_count'])).add_to(m)

    # fit to features
    if center != [0,0]:
        try:
            minx = min(x for f in feats['features'] for ring in (f['geometry']['coordinates'] if f['geometry']['type']=='Polygon' else [r for p in f['geometry']['coordinates'] for r in p]) for x,_ in ring)
            maxx = max(x for f in feats['features'] for ring in (f['geometry']['coordinates'] if f['geometry']['type']=='Polygon' else [r for p in f['geometry']['coordinates'] for r in p]) for x,_ in ring)
            miny = min(y for f in feats['features'] for ring in (f['geometry']['coordinates'] if f['geometry']['type']=='Polygon' else [r for p in f['geometry']['coordinates'] for r in p]) for _,y in ring)
            maxy = max(y for f in feats['features'] for ring in (f['geometry']['coordinates'] if f['geometry']['type']=='Polygon' else [r for p in f['geometry']['coordinates'] for r in p]) for _,y in ring)
            m.fit_bounds([[miny,minx],[maxy,maxx]])
        except Exception:
            pass

    m.save(OUT_HTML)
    print('Wrote', OUT_HTML)
