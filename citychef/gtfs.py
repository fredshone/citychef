from datetime import datetime, timedelta
from halo import HaloNotebook as Halo
import os
import networkx as nx
import pandas as pd
from pyproj import Transformer


def build_gtfs(
        transit,
        name,
        out_dir,
        agency_id=0,
        agency_name='test_bus_inc',
        frequency=15,
        route_type=3,
        from_epsg="epsg:27700",
        to_epsg=None,
):
    """https://developers.google.com/transit/gtfs/examples/gtfs-feed"""

    with Halo(text='Building GTFS...', spinner='dots') as spinner:

        if to_epsg:
            transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
        else:
            transformer = None

        gtfs_dir = os.path.join(out_dir, f'gtfs_{name}')
        if not os.path.exists(gtfs_dir):
            os.mkdir(gtfs_dir)

        agency = [{
            'agency_id': agency_id,
            'agency_name': agency_name,
            'agency_url': 'NA',
            'agency_timezone': 'NA',
            'agency_phone': 'NA',
            'agency_lang': 'NA',
        }]

        stops = []
        calendar = []
        routes = []
        trips = []
        stop_times = []

        for node, data in transit.graph.nodes(data=True):
            x, y = data['pos']
            if transformer is not None:
                x, y = transformer.transform(x, y)
            stops.append({
                'stop_id': node,
                'stop_code': node,
                'stop_name': node,
                'stop_lat': x,
                'stop_lon': y,
                'wheelchair_boarding': 'yes',
                'stop_timezone': 'NA',
                'location_type': 'NA',
                'parent_station': 'NA',
                'platform_code': 'NA',
            })

        for route_id, route in enumerate(transit.routes):
            spinner.text = f'added route {route_id}'

            service_id = route_id

            calendar.append({
                'service_id': service_id,
                'monday': 1,
                'tuesday': 1,
                'wednesday': 1,
                'thursday': 1,
                'friday': 1,
                'saturday': 1,
                'sunday': 1,
                'start_date': 20190601,
                'end_date': 20201230,
            })

            routes.append({
                'route_id': route_id,
                'agency_id': agency_id,
                'route_short_name': route_id,
                'route_long_name': route_id,
                'route_type': route_type,
                'route_url': 'NA',
                'route_color': 'NA',
                'route_text_color': 'NA',
                'checkin_duration': 'NA',
            })

            start_time = datetime(year=2021, month=9, day=9, hour=5)
            end_time = datetime(year=2021, month=9, day=9, hour=22)
            interval = timedelta(minutes=frequency)
            stop_time = timedelta(seconds=60)
            leg_time = timedelta(seconds=120)

            
            trip_idx = 0
            while start_time < end_time:
                
                trip_id = f"{route_id}-{trip_idx}"
                spinner.text = f'adding trip {trip_id}, start time {start_time}'

                trips.append({
                    'route_id': route_id,
                    'service_id': service_id,
                    'trip_id': trip_id,
                    'trip_headsign': trip_id,
                    'block_id': trip_id,
                    'wheelchair_accessible': 'NA',
                    'trip_direction_name': 'NA',
                    'exceptional': 'NA',
                })

                arrival_time = start_time

                previous_node = None
                for stop_idx, node in enumerate(
                        nx.dfs_preorder_nodes(route.g, source=route.start_node)):
                    spinner.text = f'adding route {route_id} trip {trip_id} , start time {start_time}, stop {stop_idx}'
                    
                    if previous_node is not None:
                        leg_seconds = transit.network.g.edges[previous_node][node]["weight"]
                        leg_time = timedelta(seconds=leg_seconds)
                        arrival_time += leg_time

                    departure_time = arrival_time + stop_time

                    stop_times.append({
                        'trip_id': trip_id,
                        'arrival_time': arrival_time.strftime('%H:%M:%S'),
                        'departure_time': departure_time.strftime('%H:%M:%S'),
                        'stop_id': node,
                        'stop_sequence': stop_idx,
                        'stop_headsign': stop_idx,
                        'pickup_type': 0,
                        'drop_off_type': 0,
                        'shape_dist_traveled': '',
                        'timepoint': 1,
                    })

                trip_idx += 1
                start_time += interval

        agency_df = pd.DataFrame(agency)
        stops_df = pd.DataFrame(stops)
        calendar_df = pd.DataFrame(calendar)
        routes_df = pd.DataFrame(routes)
        trips_df = pd.DataFrame(trips)
        stop_times_df = pd.DataFrame(stop_times)

        agency_df.to_csv(os.path.join(gtfs_dir, 'agency.txt'), index=False)
        stops_df.to_csv(os.path.join(gtfs_dir, 'stops.txt'), index=False)
        calendar_df.to_csv(os.path.join(gtfs_dir, 'calendar.txt'), index=False)
        routes_df.to_csv(os.path.join(gtfs_dir, 'routes.txt'), index=False)
        trips_df.to_csv(os.path.join(gtfs_dir, 'trips.txt'), index=False)
        stop_times_df.to_csv(os.path.join(gtfs_dir, 'stop_times.txt'), index=False)
