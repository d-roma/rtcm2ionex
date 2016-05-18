#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
'''

@author:  D. Roma
@contact: manuel.hernandez@upc.edu
@organization: UPC-IonSAT

@summary: Class to manage IONEX files.

@version: 1.0
@change: Improved runtime by using cStringIO

@todo:  - Writer doesn't check if input data extends over current day
        - Writer doesn't process aux data
@bug: 

'''

import datetime
import os
import numpy as np
from collections import namedtuple
import logging
import cStringIO
try:
    from enum import Enum
except:
    raise ImportError("Please install enum34")

IONEX_HEADER = "     1.0            IONOSPHERE MAPS     GPS                 IONEX VERSION / TYPE\n" + \
                "{program_name:20s}{agency_name:20s}{map_date:20s}PGM / RUN BY / DATE\n" + \
                "{description}" + \
                "{year_start:6d}{month_start:6d}{day_start:6d}{hour_start:6d}{minute_start:6d}{second_start:6d}" + 24 * " " + "EPOCH OF FIRST MAP\n" + \
                "{year_end:6d}{month_end:6d}{day_end:6d}{hour_end:6d}{minute_end:6d}{second_end:6d}" + 24 * " " + "EPOCH OF LAST MAP\n" + \
                "{map_sec_interval:6d}" + 54 * " " + "INTERVAL\n" + \
                "{maps_in_file:6d}" + 54 * " " + "# OF MAPS IN FILE\n" + \
                "  {mapping_function:4s}" + 54 * " " + "MAPPING FUNCTION\n" + \
                "{elevation_cutoff:8.1f}" + 52 * " " + "ELEVATION CUTOFF\n" + \
                "{obs_used:60s}OBSERVABLES USED\n" + \
                "{n_stations:6d}" + 54 * " " + "# OF STATIONS\n" + \
                "{n_satellites:6d}" + 54 * " " + "# OF SATELLITES\n" + \
                "{base_radius:8.1f}" + 52 * " " + "BASE RADIUS\n" + \
                "{map_dimension:6d}" + 54 * " " + "MAP DIMENSION\n" + \
                "  {hgt1:6.1f}{hgt2:6.1f}{dhgt:6.1f}" + 40 * " " + "HGT1 / HGT2 / DHGT\n" + \
                "  {lat1:6.1f}{lat2:6.1f}{dlat:6.1f}" + 40 * " " + "LAT1 / LAT2 / DLAT\n" + \
                "  {lon1:6.1f}{lon2:6.1f}{dlon:6.1f}" + 40 * " " + "LON1 / LON2 / DLON\n" + \
                "{exponent:6d}" + 54 * " " + "EXPONENT\n" + \
                "{comment}" + \
                "                                                            END OF HEADER\n"

DESCRIPTION_FORMAT = "{:60s}DESCRIPTION\n"
COMMENT_FORMAT = "{:60s}COMMENT\n"

TEC_START_FORMAT = "{:6d}" + 54 * " " + "START OF TEC MAP\n"
RMS_START_FORMAT = "{:6d}" + 54 * " " + "START OF RMS MAP\n"
EPOCH_FORMAT = "{year:6d}{month:6d}{day:6d}{hour:6d}{minute:6d}{second:6d}" + \
    24 * " " + "EPOCH OF CURRENT MAP\n"
INIT_TEC_RMS_FORMAT = "  {lat:6.1f}{lon1:6.1f}{lon2:6.1f}{dlon:6.1f}{h:6.1f}" + \
    28 * " " + "LAT/LON1/LON2/DLON/H\n"
TEC_END_FORMAT = "{:6d}" + 54 * " " + "END OF TEC MAP\n"
RMS_END_FORMAT = "{:6d}" + 54 * " " + "END OF RMS MAP\n"

# Written in TEC*10**{exponent}, default exponent value is 1, therefore 0.1 TECU
# is the default unit
DATA_FORMAT = "{:5d}"

LAST_LINE = 60 * " " + "END OF FILE\n"

FILE_NAME_FORMAT = "{analysis_center:3s}{extension:1s}{doy_start:03d}{file_sequence:1s}.{year_2d:2s}I"
MAP_DATE_FORMAT = "%d-%b-%Y %H:%M"


class IonexWriter(object):
    '''
    Ionex writer class. 
    '''
    def __init__(self, map_sec_interval, file_name=None, analysis_center="upc",
                 extension="g", file_sequence="0",
                 program_name="ionex.py", agency_name="UPC-IonSAT", map_date=None,
                 description=None,
                 date_start=None, year_start=0, month_start=0, day_start=0,
                 hour_start=0, minute_start=0, second_start=0,
                 date_end=None, year_end=0, month_end=0, day_end=0,
                 hour_end=0, minute_end=0, second_end=0,
                 mapping_function="COSZ", elevation_cutoff=0.0,
                 obs_used="",
                 n_stations=0, n_satellites=0, base_radius=6371.0,
                 map_dimension=2, exponent=-1,
                 hgt1=450.0, hgt2=450.0, dhgt=0.0,
                 lat1=87.5, lat2=-87.5, dlat=-2.5,
                 lon1=-180.0, lon2=180.0, dlon=5.0, comment=None,
                 **kwargs):
        '''
        Arguments:
        - map_sec_interval: time interval in seconds between maps
        - file_name (optional): output Ionex file name. 
        - analysis_center (opt, [1]): analysis center name.
        - extension (opt, [1]): 'g' for global or 'r' regional.
        - file_sequence (opt, [1]): a number (0, 1, ...) or letter (A, B, ...).
        - program_name (opt, [2]): program name used to generate the data. Used in the header.
        - agency_name (opt, [2]): name of the agency responsible of the ionex data.
        - map_date (opt, [2]): date at which the map was generated. Can be string
            (which will not be parsed) or datetime.datetime. Defaults to now.
        - description (opt, [2]): sets the description field of the header. If directly
            set during init, it will not be parsed.
        - date_start or year/month/day/hour/minute/second_start: one of both needs
            to be set. date_start needs to be a datetime.datetime object.  
        - date_end or year/month/day/hour/minute/second_end: one of both needs
            to be set. date_start needs to be a datetime.datetime object. 
        - mapping_function (opt, [2]): mapping function used for TEC.
        - elevation_cutoff (opt, [2]): minimum elevation angle in degrees. 0 for
            unknown, 90 for altimetry
        - obs_used (opt, [2]): str less than 61 chars specifying the observables
            used for the TEC computation (blank for theoretical)
        - n_stations (opt, [2]): number of stations used for computation
        - n_satellites (opt, [2]): number of satellites used for computation
        - base_radius (opt, [2]): mean earth radius or bottom of height
        - map_dimension (opt, [2]): dimension of TEC/RMS maps: 2 or 3
        - exponent (opt): exponent used defining the unit of the values. Defaults 
            to 0.1 TECU. Used for autoscalling the input data.
        - hgt1, hgt2, dhgt: definition of an equidistant grid in height. hgt1 to hgt2 
            (both included) with increment dhgt. For map_dimension = 2, hgt1 needs to
            be equal to hgt2. In Km.
        - lat1, lat2, dlat: definition of an equidistant grid in latitude. lat1 to lat2 
            (both included) with increment dlat. In degrees.
        - lon1, lon2, dlon: definition of an equidistant grid in longitude. lon1 to lon2 
            (both included) with increment dlon. In degrees.
        - comment (opt, [2]): sets the comment field of the header. If directly
            set during init, it will not be parsed. 

        [1] used for generating the file_name, if not set.
        [2] used for generating the IONEX header 
        '''

        if type(analysis_center) is not str or len(analysis_center) != 3:
            raise AttributeError(
                "analysis_center should be a string of 3 letters")
        self.analysis_center = analysis_center.upper()

        if type(extension) is not str or len(extension) != 1:
            raise AttributeError("extension should be a string of 1 letter")
        self.extension = extension.upper()

        if type(file_sequence) is int:
            self.file_sequence = str(file_sequence)
        if type(extension) is not str or len(extension) != 1:
            raise AttributeError(
                "file_sequence should be str or int of length 1")
        else:
            self.file_sequence = file_sequence

        self.program_name = program_name
        self.agency_name = agency_name

        if map_date is None:
            self.map_date = datetime.datetime.now().strftime(MAP_DATE_FORMAT)
        else:
            if type(map_date) == str:
                self.map_date = map_date
            elif type(map_date) == datetime.datetime:
                self.map_date = map.date.strftime(MAP_DATE_FORMAT)
            else:
                raise AttributeError(
                    "map_date set with incorrect value, needs to be string or datetime.datetime instance.")

        if description is not None and len(description) > 60:
            logging.warn(
                "Description longer than 60 characters. It's the user responsibility to take care of formating or use add_description instead")
        if description is None:
            self.description = DESCRIPTION_FORMAT.format("")
        else:
            self.description = description

        if comment is not None and len(comment) > 60:
            logging.warn(
                "Comment longer than 60 characters. It's the user responsibility to take care of formating or use add_comment instead")
        if comment is None:
            self.comment = COMMENT_FORMAT.format("")
        else:
            self.comment = comment

        if year_start == 0 and month_start == 0 and day_start == 0 and date_start is None:
            raise AttributeError(
                "date_start and year, month or day both not set. Define start date")
        if date_start is not None:
            if type(date_start) is datetime.datetime:
                self.date_start = date_start
            else:
                raise AttributeError(
                    "date_start must be of type datetime.datetime")
        else:
            self.date_start = datetime.datetime(year_start, month_start, day_start,
                                                hour_start, minute_start, second_start)
        self.year_start = self.date_start.year
        self.month_start = self.date_start.month
        self.day_start = self.date_start.day
        self.hour_start = self.date_start.hour
        self.minute_start = self.date_start.minute
        self.second_start = int(self.date_start.second)
        self.doy_start = self.date_start.timetuple().tm_yday

        if year_end == 0 and month_end == 0 and day_end == 0 and date_end is None:
            raise AttributeError(
                "date_end and year, month or day both not set. Define end date")
        if date_end is not None:
            if type(date_end) is datetime.datetime:
                self.date_end = date_end
            else:
                raise AttributeError(
                    "date_end must be of type datetime.datetime")
        else:
            self.date_end = datetime.datetime(year_end, month_end, day_end,
                                              hour_end, minute_end, second_end)
        self.year_end = self.date_end.year
        self.month_end = self.date_end.month
        self.day_end = self.date_end.day
        self.hour_end = self.date_end.hour
        self.minute_end = self.date_end.minute
        self.second_end = int(self.date_end.second)
        self.doy_end = self.date_end.timetuple().tm_yday

        if type(map_sec_interval) is not int:
            raise AttributeError("map_sec_interval needs to be of type int")
        self.map_sec_interval = map_sec_interval

        if type(mapping_function) is not str or len(mapping_function) != 4:
            raise AttributeError(
                "mapping_function must be a string of length 4")
        self.mapping_function = mapping_function

        self.elevation_cutoff = elevation_cutoff

        if type(obs_used) is not str or len(obs_used) > 60:
            raise AttributeError(
                "obs_used must be a string of length less than 60")
        self.obs_used = obs_used

        self.n_stations = n_stations
        self.n_satellites = n_satellites
        self.base_radius = base_radius

        if map_dimension != 2 and map_dimension != 3:
            raise AttributeError("map_dimension should be 2 or 3")
        elif map_dimension == 3:
            # TODO: Implement 3 dimension maps
            raise NotImplementedError(
                "Currently only map_dimension 2 is supported")
        self.map_dimension = map_dimension

        if hgt1 != hgt2:
            # TODO: Implement 3 dimension maps
            raise NotImplementedError(
                "Currently only map_dimension 2 is supported, therefore hgt1 needs to be equal to hgt2")
        self.hgt1 = hgt1
        self.hgt2 = hgt2
        self.dhgt = dhgt
        self.h = self.hgt1
        self.lat1 = lat1
        self.lat2 = lat2
        self.dlat = dlat
        self.lon1 = lon1
        self.lon2 = lon2
        self.dlon = dlon

        self.maps_values_len = ((self.lon2 - self.lon1) // self.dlon) + 1
        # self.maps_lat_entries = ((self.lat2 - self.lat1) // self.dlat) + 1
        self.tec_lat_cnt = 0
        self.tec_next_lat = self.lat1
        self.next_tec_date = None
        
        self.rms_lat_cnt = 0
        self.rms_next_lat = self.lat1
        self.next_rms_date = None

        self.exponent = exponent

        if file_name is None:
            self.set_file_path(".")
        else:
            self.file_name = file_name

        self.maps_in_file = 0
        self.expected_maps_in_file = 1 + \
            ((self.date_end - self.date_start).seconds / self.map_sec_interval)
        self.tec_string = None
        self.tec_map_cnt = 0
        self.rms_string = None
        self.rms_map_cnt = 0
        self.hgt_string = None
        self.hgt_map_cnt = 0

        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.tec_map_closed = False
        self.rms_map_closed = False

    def __str__(self):
        self.maps_in_file = self.tec_map_cnt
        string_list = [IONEX_HEADER.format(**self.__dict__)]
        if self.tec_string is not None:
            if not self.tec_map_closed:
                self.tec_map_closed = True
                self.tec_string.write(TEC_END_FORMAT.format(self.tec_map_cnt))
            string_list.append(self.tec_string.getvalue())
        if self.rms_string is not None:
            if not self.rms_map_closed:
                self.rms_map_closed = True
                self.rms_string.write(RMS_END_FORMAT.format(self.rms_map_cnt))
            string_list.append(self.rms_string.getvalue())
        if self.hgt_string is not None:
            string_list.append(self.hgt_string.getvalue())
        string_list.append(LAST_LINE)
        self.string = ''.join(string_list)
        return self.string

    def set_description(self, string_list):
        '''
        Format string to ionex description
        '''
        self.description = ""
        for line in string_list:
            if len(line) > 60:
                raise AttributeError("Lines need to be of length less than 60")
            else:
                self.description += DESCRIPTION_FORMAT.format(line)
        self.description = self.description

    def set_comment(self, string_list):
        '''
        Format string to ionex comment
        '''
        self.comment = ""
        for line in string_list:
            if len(line) > 60:
                raise AttributeError("Lines need to be of length less than 60")
            else:
                self.comment += COMMENT_FORMAT.format(line)
        self.comment = self.comment

    def set_file_path(self, file_path):
        '''
        Set file_path (path) to create ionex file
        TODO: Test if file_name is valid.
        '''
        file_name = FILE_NAME_FORMAT.format(analysis_center=self.analysis_center,
                                            extension=self.extension,
                                            doy_start=self.doy_start,
                                            file_sequence=self.file_sequence,
                                            year_2d=self.date_start.strftime("%y")).lower()
        self.file_name = os.path.join(os.path.normpath(file_path), file_name)

    def write_file(self):
        try:
            with open(self.file_name, 'w') as f:
                f.write(self.__str__())
        except:
            raise

    def change_next_tec_epoch(self, date=None):
        '''
        Change to the next TEC epoch. If date is set, check that the next epoch
        set is the same than date. 
        '''
        if self.next_tec_date is None:
            self.next_tec_date = self.date_start
        else:
            self.next_tec_date += datetime.timedelta(seconds=self.map_sec_interval)
        if date != self.next_tec_date:
            logging.warn("Current date to be set is %s and expected date is %s" % 
                  (date, self.next_tec_date))
        self.next_tec_date = date
        self._tec_map_entry_epoch(self.next_tec_date)
        
    def change_next_rms_epoch(self, date=None):
        '''
        Change to the next RMS epoch. If date is set, check that the next epoch
        set is the same than date. 
        '''
        if self.next_rms_date is None:
            self.next_rms_date = self.date_start
        else:
            self.next_rms_date += datetime.timedelta(seconds=self.map_sec_interval)
        if date != self.next_rms_date:
            logging.warn("Error, current date to be set is %s and expected date is %s" % 
                  (date, self.next_rms_date))
        self._rms_map_entry_epoch(self.next_rms_date)

    def _tec_map_entry_epoch(self, date):
        '''
        Change TEC epoch of current map
        '''
        if type(date) != datetime.datetime:
            raise TypeError("Error, date needs to be of type datetime.datetime")
        if self.tec_string is None:
            self.tec_string = cStringIO.StringIO()
        else:
            if (self.lat2 + self.dlat != self.tec_next_lat):
                raise RuntimeError(
                    "Number of lat entries written and expect don't match")
            self.tec_next_lat = self.lat1
            self.tec_string.write(TEC_END_FORMAT.format(self.tec_map_cnt))
        self.tec_map_cnt += 1
        self.tec_string.write(TEC_START_FORMAT.format(self.tec_map_cnt))
        self.tec_string.write(EPOCH_FORMAT.format(year=date.year, month=date.month,
                                               day=date.day, hour=date.hour,
                                               minute=date.minute,
                                               second=date.second))
        self.tec_map_closed = False

    def _rms_map_entry_epoch(self, date):
        '''
        Change RMS epoch of current map
        '''
        if type(date) != datetime.datetime:
            raise TypeError("Error, date needs to be of type datetime.datetime")
        if self.rms_string is None:
            self.rms_string = cStringIO.StringIO()
        else:
            if (self.lat2 + self.dlat != self.rms_next_lat):
                raise RuntimeError(
                    "Number of lat entries written and expect don't match")
            self.rms_next_lat = self.lat1
            self.rms_string.write(RMS_END_FORMAT.format(self.rms_map_cnt))
        self.rms_map_cnt += 1
        self.rms_string.write(RMS_START_FORMAT.format(self.rms_map_cnt))
        self.rms_string.write(EPOCH_FORMAT.format(year=date.year, month=date.month,
                                               day=date.day, hour=date.hour,
                                               minute=date.minute,
                                               second=date.second))
        self.rms_map_closed = False    

    def add_tec_entry(self, tec_array, lat_vec, date=None):
        '''
        Change to the next TEC epoch and set all the TEC entries.
        '''
        self.change_next_tec_epoch(date)
        for i, entry in enumerate(tec_array):
            self.add_tec_lat_entry(lat_vec[i], entry)
            # self._add_tec_entry(tec_array)
        return self.close_tec()
    
    def add_rms_entry(self, rms_array, lat_vec, date=None):
        '''
        Change to the next RMS epoch and set all the RMS entries.
        '''
        self.change_next_rms_epoch(date)
        for i, entry in enumerate(rms_array):
            self.add_rms_lat_entry(lat_vec[i], entry)
        return self.close_rms()

    def add_tec_lat_entry(self, lat, tec_array):
        '''
        Add next TEC latitude entry for current map. 
        '''
        if type(tec_array) is not np.ndarray:
            raise TypeError("tec_array must be of type numpy.ndarray")
        if len(tec_array) != self.maps_values_len:
            raise ValueError("Error, tec array length needs to be %d and it was %d." % 
                  (self.maps_values_len, len(tec_array)))
        if lat != self.tec_next_lat:
            logging.warn("Got tec entry for lat %d, expected for lat %d" % 
                  (lat, self.tec_next_lat))
        self._add_tec_entry(tec_array)

    def add_rms_lat_entry(self, lat, rms_array):
        '''
        Add next RMS latitude entry for current map.
        '''
        if type(rms_array) is not np.ndarray:
            raise TypeError("rms_array must be of type numpy.ndarray")
        if len(rms_array) != self.maps_values_len:
            raise ValueError("Error, rms array length needs to be %d and it was %d." % 
                  (self._values_len, len(rms_array)))
        if lat != self.rms_next_lat:
            logging.warn("Got rms entry for lat %d, expected for lat %d" % 
                  (lat, self.rms_next_lat))
        self._add_rms_entry(rms_array)

    def _add_tec_entry(self, tec_array):
        '''
        Add next TEC latitude entry for current map
        '''
        self.tec_string.write(INIT_TEC_RMS_FORMAT.format(lat=self.tec_next_lat,
                                                      lon1=self.lon1,
                                                      lon2=self.lon2,
                                                      dlon=self.dlon,
                                                      h=self.h))
        self.tec_next_lat += self.dlat
        tec_array = (tec_array * 10 ** np.abs(self.exponent)).astype('int', copy=False)
        stop = 16 
        for start in range(0, tec_array.size, 16):
            sub_arr = tec_array[start:stop]
            self.tec_string.write((sub_arr.size * DATA_FORMAT + "\n").format(*sub_arr))
            stop += 16

    def _add_rms_entry(self, rms_array):
        '''
        Add next TEC latitude entry for current map
        '''
        self.rms_string.write(INIT_TEC_RMS_FORMAT.format(lat=self.rms_next_lat,
                                                      lon1=self.lon1,
                                                      lon2=self.lon2,
                                                      dlon=self.dlon,
                                                      h=self.h))
        self.rms_next_lat += self.dlat
        rms_array = (rms_array * 10 ** np.abs(self.exponent)).astype('int', copy=False)
        stop = 16 
        for start in range(0, rms_array.size, 16):
            sub_arr = rms_array[start:stop]
            self.rms_string.write((sub_arr.size * DATA_FORMAT + "\n").format(*sub_arr))
            stop += 16

    def close_tec(self):
        '''
        Returns true and close last tec map if all maps for all the epochs have been written. 
        '''
        if (self.lat2 == (self.tec_next_lat - self.dlat)):
            self.tec_string.write(TEC_END_FORMAT.format(self.tec_map_cnt))
            self.tec_map_closed = True
            if (self.next_tec_date >= self.date_end and
                self.tec_map_cnt != self.expected_maps_in_file):
                    logging.error("Expected number of TEC maps was %d, number of TEC maps written is %d" % 
                                  (self.expected_maps_in_file, self.tec_map_cnt))
            return True
        return False
        
    def close_rms(self):
        '''
        Returns true and close last RMS map if all maps for all the epochs have been written. 
        '''
        if (self.lat2 == (self.rms_next_lat - self.dlat)):
            self.rms_string.write(RMS_END_FORMAT.format(self.rms_map_cnt))
            self.rms_map_closed = True
            if (self.next_tec_date >= self.date_end and
                self.rms_map_cnt != self.expected_maps_in_file):
                    logging.error("Expected number of RMS maps was %d, number of RMS maps written is %d" % 
                                  (self.expected_maps_in_file, self.rms_map_cnt))
            return True
        return False
        
_ionex_header_fields = namedtuple('_ionex_header_fields', ['version',
                                                           'type',
                                                           'satellite_system',
                                                           'program_name',
                                                           'agency',
                                                           'creation_date',
                                                           'description',
                                                           'comment',
                                                           'epoch_first_map',
                                                           'epoch_last_map',
                                                           'interval',
                                                           'n_maps',
                                                           'mapping',
                                                           'ele_cutoff',
                                                           'obs_used',
                                                           'n_stations',
                                                           'n_satellites',
                                                           'earth_radius',
                                                           'map_dim',
                                                           'hgt1', 'hgt2', 'dhgt',
                                                           'lat1', 'lat2', 'dlat',
                                                           'lon1', 'lon2', 'dlon',
                                                           'exponent', 'aux_data'                                                        
                                                           ])

_map = namedtuple('_map', 'id epoch value')

_fsm_reader = Enum('_fsm_reader', 'header map_type read_map')
_entry_type = Enum('_entry_type', 'tec rms hgt')

def_header_dict = {'version': 1.0,
                   'type': 'I',
                   'satellite_system': 'GPS',
                   'program_name': None,
                   'agency': None,
                   # Standard doesn't define the date format
                   # 'creation_date': datetime.datetime.now(),
                   'creation_date': '',
                   'description': None,
                   'comment': None,
                   'epoch_first_map': datetime.datetime.now(),
                   'epoch_last_map': datetime.datetime.now(),
                   'interval': 900,
                   'n_maps': 97,
                   'mapping': 'COSZ',
                   'ele_cutoff': 0.0,
                   'obs_used': '',
                   'n_stations': 0,
                   'n_satellites': 0,
                   'earth_radius': 6371.0,
                   'map_dim': 2,
                   'hgt1': 400.0, 'hgt2': 400.0, 'dhgt': 0.0,
                   'lat1':-87.5, 'lat2':87.5, 'dlat':2.5,
                   'lon1':-180.0, 'lon2':180.0, 'dlon':5.0,
                   'exponent':-1,
                   'aux_data': None}

class IonexReader(object):
    '''
    Ionex reader class
    '''
    def __init__(self, file_path, strict_mode=False,
                 header_dict=def_header_dict):
        '''
        :file_path: path to the ionex file to read
        :strict_mode: error if not complaint with the standard
        '''
        state = _fsm_reader.header
        header_fields_cnt = 0
        if not strict_mode:
            exp_len_map_entry = int((header_dict['lon2'] - 
                                     header_dict['lon1']) / 
                                     header_dict['dlon']) + 1
        aux_data = False
        self.tec_maps = []
        self.rms_maps = []
        self.hgt_maps = []
        self.tec_map_next_cnt = 0
        self.rms_map_next_cnt = 0
        self.hgt_map_next_cnt = 0
        with open(file_path, 'r') as f:
            for line in f:
                line_type = line[60:]
                if state == _fsm_reader.header:
                    # Using in allows ignoring whitespaces before end of line and 
                    #  header displaced outside expected position
                    if aux_data:
                        if "END OF AUX DATA" in line_type:
                            aux_data = False
                        else:
                            header_dict['aux_data'].append(line) 
                    elif "IONEX VERSION / TYPE" in line_type: 
                        header_dict['version'] = float(line[:8])
                        header_dict['type'] = line[20:21]
                        header_dict['satellite_system'] = line[40:43]
                        header_fields_cnt += 1
                    elif "PGM / RUN BY / DATE" in line_type:
                        header_dict['program_name'] = line[:20]
                        header_dict['agency'] = line[20:40]
                        header_dict['creation_date'] = line[40:60]
                        header_fields_cnt += 1
                    elif "DESCRIPTION" in line_type:
                        if header_dict['description'] is None:
                            header_dict['description'] = [line[:60]]
                        else:
                            header_dict['description'].append(line[:60])
                    elif "COMMENT" in line_type:
                        if header_dict['comment'] is None:
                            header_dict['comment'] = [line[:60]]
                        else:
                            header_dict['comment'].append(line[:60])
                    elif "EPOCH OF FIRST MAP" in line_type:
                        ll = line[:60].split()
                        header_dict['epoch_first_map'] = datetime.datetime(year=int(ll[0]), \
                                                                           month=int(ll[1]), \
                                                                           day=int(ll[2]), \
                                                                           hour=int(ll[3]), \
                                                                           minute=int(ll[4]), \
                                                                           second=int(ll[5]))
                        header_fields_cnt += 1
                    elif "EPOCH OF LAST MAP" in line_type:
                        ll = line[:60].split()
                        header_dict['epoch_last_map'] = datetime.datetime(year=int(ll[0]), \
                                                                           month=int(ll[1]), \
                                                                           day=int(ll[2]), \
                                                                           hour=int(ll[3]), \
                                                                           minute=int(ll[4]), \
                                                                           second=int(ll[5]))
                        header_fields_cnt += 1
                    elif "INTERVAL" in line_type:
                        header_dict['interval'] = int(line[:6])
                        header_fields_cnt += 1
                    elif "# OF MAPS IN FILE" in line_type:
                        header_dict['n_maps'] = int(line[:6])
                        header_fields_cnt += 1
                    elif "MAPPING FUNCTION" in line_type:
                        header_dict['mapping'] = line[2:6]
                        header_fields_cnt += 1
                    elif "ELEVATION CUTOFF" in line_type:
                        header_dict['ele_cutoff'] = float(line[:8])
                        header_fields_cnt += 1
                    elif "OBSERVABLES USED" in line_type:
                        header_dict['obs_used'] = line[:60]
                        header_fields_cnt += 1
                    elif "# OF STATIONS" in line_type:
                        header_dict['n_stations'] = int(line[:6])
                    elif "# OF SATELLITES" in line_type:
                        header_dict['n_satellites'] = int(line[:6])
                    elif "BASE RADIUS" in line_type:
                        header_dict['earth_radius'] = float(line[:8])
                        header_fields_cnt += 1
                    elif "MAP DIMENSION" in line_type:
                        header_dict['map_dim'] = int(line[:6])
                        header_fields_cnt += 1
                    elif "HGT1 / HGT2 / DHGT" in line_type:
                        header_dict['hgt1'] = float(line[2:8])
                        header_dict['hgt2'] = float(line[8:14])
                        header_dict['dhgt'] = float(line[14:20])
                        header_fields_cnt += 1
                    elif "LAT1 / LAT2 / DLAT" in line_type:
                        header_dict['lat1'] = float(line[2:8])
                        header_dict['lat2'] = float(line[8:14])
                        header_dict['dlat'] = float(line[14:20])
                        header_fields_cnt += 1
                    elif "LON1 / LON2 / DLON" in line_type:
                        header_dict['lon1'] = float(line[2:8])
                        header_dict['lon2'] = float(line[8:14])
                        header_dict['dlon'] = float(line[14:20])
                        header_fields_cnt += 1   
                        exp_len_map_entry = int((header_dict['lon2'] - 
                                                 header_dict['lon1']) / 
                                                 header_dict['dlon']) + 1
                    elif "EXPONENT" in line_type:
                        if strict_mode:
                            header_dict['exponent'] = int(line[:6])
                        else:
                            header_dict['exponent'] = int(line[:60])
                    elif "START OF AUX DATA" in line_type:
                        aux_data = True
                        header_dict['aux_data'] = []
                    elif "END OF HEADER" in line_type:
                        self.header = _ionex_header_fields(**header_dict)
                        header_dict = None
                        state = _fsm_reader.map_type
                        if strict_mode:
                            assert header_fields_cnt >= 14, \
                                "Missing some required IONEX fields"
                        elif header_fields_cnt < 14:
                            logging.warn("Missing some required IONEX fields")
                    else:
                        logging.error("Unknown type of ionex header entry:\n%s" % line[:-1])
                elif state == _fsm_reader.map_type:
                    entry = {'id': int(line[:6])}
                    state = _fsm_reader.read_map
                    line_cnt = 0
                    if "START OF TEC MAP" in line_type:
                        entry_type = _entry_type.tec
                    elif "START OF RMS MAP" in line_type:
                        entry_type = _entry_type.rms
                    elif "START OF HEIGHT MAP" in line_type:
                        entry_type = _entry_type.hgt
                    elif "END OF FILE" in line_type:
                        logging.debug("Reached end of file")
                    else:
                        raise RuntimeError("Unknown type of map\n%s" % line[:-1])
                elif state == _fsm_reader.read_map:
                    if "EPOCH OF CURRENT MAP" in line_type:
                        assert line_cnt == 0, "Unexpected position of epoch"
                        line_cnt += 1
                        ll = line[:60].split()
                        hour = int(ll[3])
                        if hour == 24:
                            hour = 23
                            minute = 59
                            second = 59
                        else:
                            minute = int(ll[4])
                            second = int(float(ll[5]))
                        entry['epoch'] = datetime.datetime(year=int(ll[0]), \
                                                           month=int(ll[1]), \
                                                           day=int(ll[2]), \
                                                           hour=hour, \
                                                           minute=minute, \
                                                           second=second)  
                        lat_cnt = 0
                        entry['value'] = []
                        c_values = []
                    elif "LAT/LON1/LON2/DLON/H" in line_type:
                        c_lat = float(line[2:8])
                        c_lon1 = float(line[8:14])
                        c_lon2 = float(line[14:20])
                        c_dlon = float(line[20:26])
                        c_h = float(line[26:32])
                        assert line_cnt == 1, "Unexpected position of lat/lon/..."
                        assert c_lat == self.header.lat1 + lat_cnt * self.header.dlat, \
                          ("Expected lat %f, found %f" % 
                           (c_lat, self.header.lat1 + lat_cnt * self.header.dlat))
                        assert c_lon1 == self.header.lon1, "lon1 error"
                        assert c_lon2 == self.header.lon2, "lon2 error"
                        assert c_dlon == self.header.dlon, "dlon error"
                        # TODO: Currently only 2-d maps are accepted
                        assert c_h == self.header.hgt1, "height error"
                        if lat_cnt != 0:
                            assert len(c_values) == exp_len_map_entry, \
                                ("Expected %d values, found %d" % \
                                 (len(c_values), exp_len_map_entry))
                            entry['value'].append(np.array(c_values) * 
                                              10 ** self.header.exponent)
                            c_values = []
                        lat_cnt += 1
                    elif "END OF TEC MAP" in line_type:
                        assert entry['id'] == int(line[:6]), \
                            "End of TEC map ID doesn't match"
                        assert c_lat == self.header.lat2, \
                            "Missing some lat values for current map"
                        assert entry_type == _entry_type.tec, \
                            "End of tec map found, expected another kind of map"
                        assert len(c_values) == exp_len_map_entry, \
                                ("Expected %d values, found %d" % \
                                 (len(c_values), exp_len_map_entry))
                        entry['value'].append(np.array(c_values) * 
                                              10 ** self.header.exponent)
                        self.tec_maps.append(_map(**entry))
                        state = _fsm_reader.map_type
                    elif "END OF RMS MAP" in line_type:
                        assert entry['id'] == int(line[:6]), \
                            "End of RMS map ID doesn't match"
                        assert c_lat == self.header.lat2, \
                            "Missing some lat values for current map"
                        assert entry_type == _entry_type.rms, \
                            "End of RMS map found, expected another kind of map"
                        assert len(c_values) == exp_len_map_entry, \
                                ("Expected %d values, found %d" % \
                                 (len(c_values), exp_len_map_entry))
                        entry['value'].append(np.array(c_values) * 
                                              10 ** self.header.exponent)
                        self.rms_maps.append(_map(**entry))
                        state = _fsm_reader.map_type
                    elif "END OF HGT MAP" in line_type:
                        assert entry['id'] == int(line[:6]), \
                            "End of HGT map ID doesn't match"
                        assert c_lat == self.header.lat2, \
                            "Missing some lat values for current map"
                        assert entry_type == _entry_type.hgt, \
                            "End of hgt map found, expected another kind of map"
                        assert len(c_values) == exp_len_map_entry, \
                                ("Expected %d values, found %d" % \
                                 (len(c_values), exp_len_map_entry))
                        entry['value'].append(np.array(c_values) * 
                                              10 ** self.header.exponent)
                        self.hgt_maps.append(_map(**entry))
                        state = _fsm_reader.map_type
                    else:
                        prev_pos = 0
                        for cur_pos in range(5, len(line) + 1, 5):
                            c_values.append(int(line[prev_pos:cur_pos]))
                            prev_pos = cur_pos
        if strict_mode:
            assert self.header.n_maps == len(self.tec_maps), \
                ("Expected number of maps (%d) and maps read (%d) doesn't match" % \
                  (self.header.n_maps, len(self.tec_maps)))
        else:
            if self.header.n_maps != len(self.tec_maps):
                logging.warn("Expected number of maps (%d) and maps read (%d) doesn't match" % \
                  (self.header.n_maps, len(self.tec_maps)))
                
    def get_header(self):
        return self.header
    
    def get_tec_maps(self):
        return self.tec_maps
    
    def get_next_tec_map(self):
        self.tec_map_next_cnt += 1
        return self.tec_maps[self.tec_map_next_cnt - 1]
    
    def set_next_tec_map_cnt(self, prm):
        if prm > len(self.tec_maps):
            raise ValueError("tec_map_cnt needs to be lower than %d" % 
                             len(self.tec_maps))
        else:
            self.tec_map_next_cnt = prm
