from lxml import etree as et
import os
import pandas as pd

import gzip
import halo

def Halo(*args, **kw):
    ipython = False
    try:
        get_ipython()
        ipython = True
    except NameError:
        pass
    if ipython:
        return halo.HaloNotebook(*args, **kw)
    return halo.Halo(*args, **kw)

def is_xml(location):
    return location.lower().endswith(".xml")


def is_gzip(location):
    return location.lower().endswith(".gz") or location.lower().endswith(".gzip")


def create_local_dir(directory):
    if directory != "" and not os.path.exists(directory):
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
    with Halo(text='Building OSM network xml...', spinner='dots') as spinner:

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
                    'id': str(node),
                    'uid': str(node),
                    'lat': str(data['pos'][1]),
                    'lon': str(data['pos'][0]),
                    'version': "6",
                    'timestamp': '2016-07-31T12:34:30Z',
                    'user': 'test',
                    'changeset': '41144045',
                }
            )
        spinner.text = 'added all nodes'

        for idx, (u, v, d) in enumerate(g.edges(data=True)):
            index = f"00{idx}"
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
                    'ref': str(u),
                }
            )
            nd_element = et.SubElement(
                way_element,
                'nd',
                {
                    'ref': str(v),
                }
            )
            tag_element = et.SubElement(
                way_element,
                'tag',
                {
                    'k': d.get("label", ("unknown", "unknown"))[0],
                    'v': d.get("label", ("unknown", "unknown"))[1],
                }
            )

        spinner.text = 'added all ways'

        write_content(osm, location=path)
        spinner.succeed('done')


