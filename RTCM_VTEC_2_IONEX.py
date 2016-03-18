#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
'''

@author:  D. Roma
@contact: manuel.hernandez@upc.edu
@organization: UPC-IonSAT

@summary: RTCM VTEC entries converter to IONEX format 

@version: 0.1
@change: Initial version

@todo:    - If more than one day is present in the input RTCM VTEC file, it will
            go inside the same IONEX file. 
@bug: 

'''

from collections import deque
import datetime
import os
import argparse

try:
    import numpy as np
except ImportError:
    raise ImportError("Missing numpy library")

try:
    from scipy import special
    from scipy.misc import factorial
except ImportError:
    raise ImportError("Missing scipy library")

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Missing matplotlib library")

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    raise ImportError("Missing mpl_toolkit library")

try:
    import progressbar
except ImportError:
    raise ImportError("Missing progressbar library")

# try:
#     from astropy.time import Time
# except ImportError:
#     raise ImportError("Missing astropy library")


from ionex import IonexWriter


def kron_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def vtec_sph_harm(longitude, latitude, cos_coeff, sin_coeff, gps_time_s):
    # s = (longitude + gps_time_s * 15.0 / 3600) * np.pi / 180.0
    s = np.mod((longitude*np.pi/180) + ((gps_time_s - 2*60*60)*np.pi/43200), 
               2*np.pi)

    #lat = (90-latitude)*np.pi/180
    lat = (latitude) * np.pi / 180

    vtec = 0
    for n in range(cos_coeff.shape[0]):
        for m in range(cos_coeff.shape[1]):
            A = np.sqrt(factorial(n - m, exact=True) * (2 * n + 1)
                        * (2 - kron_delta(0, m)) / factorial(n + m, exact=True))
            vtec += A * special.lpmv(m, n, np.sin(lat)) * \
                (cos_coeff[n, m] * np.cos(m * s) +
                 sin_coeff[n, m] * np.sin(m * s))

    return vtec


def RTCM_VTEC_reader(file_name):
    '''
    > VTEC YEAR MONTH DAY HOUR MINUTE SECONDS SSR_X #Records MOUNTPOINT
    layer_number max_degree max_order height_layer
    C
    0 --- degree --- max_degree
    |
    order
    |
    max_order
    S
    0 --- degree --- max_degree
    |
    order
    |
    max_order

    '''
    vtec_entries = deque()

    start_found = 0
    for line in file_name:
        in_line = line.split()
        #print(in_line, len(in_line))
        # Ignore blank lines
        if len(in_line) < 1:
            continue
        # Header line, update date time information
        elif in_line[0] == '>':
            #gpsweek = int(in_line[1])
            #gpsweek_sec = float(in_line[2])
            #(year, month, day, hour, minute, sec) = dateConversion(gpsweek, \
            #                                                       gpsweek_sec)
            if start_found == 2:
                # Do what ever with the previous entry
                assert cnt_order == 2 * \
                    vtec_record['n_order'], "ERROR. cnt_order is %d, should be %d" % (
                        cnt_order, 2 * vtec_record['n_order'])
                vtec_record['C'] = C
                vtec_record['S'] = S
                vtec_entries.append(vtec_record)
            vtec_record = {}
            vtec_record['type'] = in_line[1]
            vtec_record['year'] = int(in_line[2])
            vtec_record['month'] = int(in_line[3])
            vtec_record['day'] = int(in_line[4])
            vtec_record['hour'] = int(in_line[5])
            vtec_record['minute'] = int(in_line[6])
            vtec_record['seconds'] = float(in_line[7])
            vtec_record['ssr_x'] = int(in_line[8])
            vtec_record['n_records'] = int(in_line[9])
            vtec_record['mountpoint'] = in_line[10]
            vtec_record['datetime'] = datetime.datetime(vtec_record['year'],
                                                        vtec_record['month'],
                                                        vtec_record['day'],
                                                        vtec_record['hour'],
                                                        vtec_record['minute'],
                                                        int(vtec_record['seconds']))
            if vtec_record['type'] != 'VTEC':
                start_found = 0
                print("Error, wrong entry type. Expecting VTEC, found %s" %
                      vtec_record['type'])
            cnt_order = 0
            start_found = 1
        elif start_found == 1:
            vtec_record['n_layer'] = int(in_line[0])
            # Order and degree go from 0 to value, therefore we add 1 (value+1
            # elements)
            vtec_record['n_degree'] = int(in_line[1]) + 1
            vtec_record['n_order'] = int(in_line[2]) + 1
            vtec_record['height_layer'] = float(in_line[3])
            C = np.empty([vtec_record['n_order'], vtec_record['n_degree']])
            S = np.empty([vtec_record['n_order'], vtec_record['n_degree']])
            start_found = 2
        elif start_found == 2:
            if cnt_order < vtec_record['n_order']:
                C[cnt_order, :] = list(map(float, in_line))
            else:
                S[cnt_order - vtec_record['n_order'],
                    :] = list(map(float, in_line))
            cnt_order += 1

    return vtec_entries

if __name__ == '__main__':

    desc = "Generate a IONEX file from a one-day RTCM VTEC file."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-i', '--rtcm-file', default='VTEC0120.16C',
                        type=argparse.FileType('r'),
                        help='Absolute path to the RTCM VTEC input file',
                        required=True, dest='file_name')
    parser.add_argument('--dtsec-file', default=15*60,  # 15*60
                        type=int,
                        help='Time between two output tec maps of the IONEX file',
                        required=False, dest='dT_sec_file')
    parser.add_argument('--dtsec-map', default=1*60*60,  # 1*60*60
                        type=int,
                        help='Time between two output tec image maps. 0 to disable image generation',
                        required=False, dest='dT_sec_map')
    parser.add_argument('--lat1', default=87.5, type=float,
                        help='Mininum value of the latitude used for the maps and IONEX file',
                        required=False, dest='min_lat')
    parser.add_argument('--lat2', default=-87.5, type=float,
                        help='Maximum value of the latitude used for the maps and IONEX file',
                        required=False, dest='max_lat')
    parser.add_argument('--dlat', default=-2.5, type=float,
                        help='Step for the latitude used for the maps and IONEX file',
                        required=False, dest='dlat')
    parser.add_argument('--lon1', default=-180, type=float,
                        help='Mininum value of the longitude used for the maps and IONEX file',
                        required=False, dest='min_lon')
    parser.add_argument('--lon2', default=180, type=float,
                        help='Maximum value of the longitude used for the maps and IONEX file',
                        required=False, dest='max_lon')
    parser.add_argument('--dlon', default=5.0, type=float,
                        help='Step for the longitude used for the maps and IONEX file',
                        required=False, dest='dlon')

    args = parser.parse_args()

#     file_name = "VTEC0120.16C"
#     #dT_sec_map = 6 * 60 * 60
#     dT_sec_map = 1 * 60 * 60
#     dT_sec_ionex = 15 * 60
#     #dT_sec_ionex = 6 * 60 * 60
#     dlat = -2.5
#     min_lat = 87.5
#     max_lat = -87.5
#     min_lon = -180
#     dlon = 5.0
#     max_lon = 180

    file_name = args.file_name
    dT_sec_map = args.dT_sec_map
    dT_sec_ionex = args.dT_sec_file
    dlat = args.dlat
    min_lat = args.min_lat
    max_lat = args.max_lat
    min_lon = args.min_lon
    dlon = args.dlon
    max_lon = args.max_lon

    dT_sec_map = datetime.timedelta(seconds=dT_sec_map)
    dT_sec_ionex = datetime.timedelta(seconds=dT_sec_ionex)

    WORLD_FIG_SIZE = [16, 12]
    widgets = [progressbar.Percentage(), ' ',
               progressbar.Bar(marker=progressbar.RotatingMarker()),
               ' ', progressbar.ETA()]

    if not os.path.exists("./plots"):
        os.mkdir("./plots")

    lat_vec = np.arange(min_lat, max_lat + dlat, dlat)
    lon_vec = np.arange(min_lon, max_lon + dlon, dlon)
    lon, lat = np.meshgrid(lon_vec, lat_vec)

    q = RTCM_VTEC_reader(file_name)
    file_name.close()

    last_entry_map = datetime.datetime(1, 1, 1)
    last_entry_ionex = datetime.datetime(1, 1, 1)

    ionexWriter = IonexWriter(dT_sec_ionex.seconds, analysis_center="CNS",
                              extension="g", file_sequence="0",
                              program_name="ionex.py", agency_name="CNES",
                              date_start=q[0]['datetime'], 
                              date_end=q[-1]['datetime'],
                              hgt1=q[0]['height_layer'] / 1000, 
                              hgt2=q[0]['height_layer'] / 1000, dhgt=0.0,
                              lat1=min_lat, lat2=max_lat, dlat=dlat,
                              lon1=min_lon, lon2=max_lon, dlon=dlon)

    gps_start_date = datetime.datetime(1970, 1, 6)

    bar = progressbar.ProgressBar(
        maxval=ionexWriter.expected_tec_maps_in_file, widgets=widgets).start()

    for entry in q:
        #TODO: Check if day has changed to close current ionex file and generate a new one. 
        
        if (entry['datetime'] - last_entry_map >= dT_sec_map and dT_sec_map != 0):
            gen_map = True
            last_entry_map = entry['datetime']
        else:
            gen_map = False

        if (entry['datetime'] - last_entry_ionex >= dT_sec_ionex):
            gen_ionex = True
            last_entry_ionex = entry['datetime']
        else:
            gen_ionex = False

        if gen_ionex or gen_map:
            #univ_time_s = Time(entry['datetime'], scale='utc')
            #gps_time_s = (univ_time_s.gps % 86400)
            gps_time_s = (entry['datetime'] - gps_start_date).total_seconds() % 86400
            vtec = vtec_sph_harm(longitude=lon, latitude=lat, cos_coeff=entry['C'],
                                 sin_coeff=entry['S'], gps_time_s=gps_time_s)

        if gen_map:
            fig = plt.figure(figsize=WORLD_FIG_SIZE)
            m = Basemap(llcrnrlon=np.min([min_lon, max_lon]),
                        llcrnrlat=np.min([min_lat, max_lat]),
                        urcrnrlon=np.max([min_lon, max_lon]),
                        urcrnrlat=np.max([min_lat, max_lat]),
                        lon_0=0, lat_0=0, resolution='l')
            m.drawcoastlines(linewidth=0.5, color='black')
            m.drawparallels(np.arange(-90, +91, 30), labels=[1, 1, 0, 1])
            m.drawmeridians(np.arange(-180, 181, 60), labels=[1, 1, 0, 1])
            cs = m.pcolormesh(lon, lat, vtec,  cmap=plt.get_cmap('rainbow'))
            cbar = m.colorbar(cs, location='bottom', pad="5%")
            cbar.set_label("TECU")
            plt.title(
                "VTEC {mountpoint} {year:04d}-{month:02d}-{day:02d} @ {hour:02d}:{minute:02d}:{seconds:05.3f}".format(**entry))
            plt.tight_layout(pad=4)
            plt.savefig("./plots/VTEC_{mountpoint}_{year:04d}{month:02d}{day:02d}_{hour:02d}.png".format(**entry),
                        dpi=300, bbox_inches='tight')
            plt.close()

        if gen_ionex:
            ionexWriter.add_tec_entry(vtec, lat_vec, entry['datetime'])
            bar.update(ionexWriter.tec_map_cnt)

    ionexWriter.write_file()
    bar.finish()
