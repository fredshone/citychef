from halo import HaloNotebook as Halo
from lxml import etree as et
import os
import pandas as pd
import gzip


def is_xml(location):
    return location.lower().endswith(".xml")


def is_gzip(location):
    return location.lower().endswith(".gz") or location.lower().endswith(".gzip")


def create_local_dir(directory):
    if not os.path.exists(directory):
        print('Creating {}'.format(directory))
        os.makedirs(directory)


def xml_tree(content):
    tree = et.tostring(content,
                       pretty_print=True,
                       xml_declaration=False,
                       encoding='UTF-8')
    return tree


def xml_content(content):
    xml_version = b'<?xml version="1.0" encoding="UTF-8"?>'
    tree = xml_tree(content)
    return xml_version + tree


def write_content(content, location, **kwargs):
    with Halo(text="\tWriting output to local file system at {}".format(location),
              spinner='dots') as spinner:
        create_local_dir(os.path.dirname(location))
        if isinstance(content, pd.DataFrame):
            content.to_csv(location)
        else:
            if is_xml(location):
                content = xml_content(content, **kwargs)
            if is_gzip(location):
                file = gzip.open(location, "w")
            else:
                file = open(location, "wb")
            file.write(content)

            spinner.succeed('Content written to {}'.format(location))
            file.close()


def nx_to_osm(g, path):
    with Halo(text='Building attributes xml...', spinner='dots') as spinner:

        osm = et.Element('osm', {'version': '0.6', 'generator': 'JOSM'})  # start forming xml
        bounds_element = et.SubElement(
            osm,
            'bounds',
            {'minlat': '-1', 'minlon': '-1', 'maxlat': '1', 'maxlon': '1', 'origin': 'test'},
        )

        for node, data in g.nodes(data=True):
            node_element = et.SubElement(
                osm,
                'node',
                {
                    'id': node,
                    'uid': node,
                    'lat': str(data['pos'][1]),
                    'lon': str(data['pos'][0]),
                    'version': "6",
                    'timestamp': '2016-07-31T12:34:30Z',
                    'user': 'test',
                    'changeset': '41144045',
                }
            )
        spinner.text = 'added all nodes'

        for idx, (u, v) in enumerate(g.edges()):
            index = f"00{idx}00"
            way_element = et.SubElement(
                osm,
                'way',
                {
                    'id': index,
                    'uid': index,
                    'lat': str(data['pos'][1]),
                    'lon': str(data['pos'][0]),
                    'version': "6",
                    'timestamp': '2016-07-31T12:34:30Z',
                    'user': 'test',
                    'changeset': '41144045',
                }
            )
            nd_element = et.SubElement(
                way_element,
                'nd',
                {
                    'ref': u,
                }
            )
            nd_element = et.SubElement(
                way_element,
                'nd',
                {
                    'ref': v,
                }
            )
            tag_element = et.SubElement(
                way_element,
                'tag',
                {
                    'k': 'highway',
                    'v': 'unclassified',
                }
            )

        spinner.text = 'added all ways'

        write_content(osm, location=path)
        spinner.succeed('done')
