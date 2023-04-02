import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [10, 5]
from matplotlib import cm
import os

def C8_3d_plots(PATH, title=""):
    tracks = pd.read_parquet(PATH)
    print(f"Max energy (kinetic): {max(tracks['start_energy'])} GeV, Min energy (kinetic): {min(tracks['end_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['start_energy'])}")

    # calculate normalized time
    max_t = max(tracks['end_t'])
    min_t = min(tracks['start_t'])
    norm_t = (tracks['start_t']-min_t)/max_t 

    current_shower = -1
    for pdg, x1, x2, y1, y2, z1, z2, t, shower in zip(tracks['pdg'], 
                                       tracks['start_x'], tracks['end_x'],
                                       tracks['start_y'], tracks['end_y'],
                                       tracks['start_z'], tracks['end_z'],
                                         norm_t, tracks['shower']):
        if (current_shower!=shower):
            # new shower -> new plot 
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_xlabel('x / m')
            ax.set_ylabel('y / m')
            ax.set_zlabel('z / m')
            current_shower = shower
        ax.plot([x1, x2], [y1, y2], [z1-6371000, z2-6371000], c=cm.hot(np.abs(t)))
        plt.title(title)

def C8_zx_plots(PATH, title=""):
    tracks = pd.read_parquet(PATH)
    print(f"Max energy (kinetic): {max(tracks['start_energy'])} GeV, Min energy (kinetic): {min(tracks['end_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['start_energy'])}")
    max_t = max(tracks['end_t'])
    min_t = min(tracks['start_t'])
    norm_t = (tracks['start_t']-min_t)/max_t 
    
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    current_shower = 0
    for pdg, x1, x2, y1, y2, z1, z2, t, shower in zip(tracks['pdg'], 
                                           tracks['start_x'], tracks['end_x'],
                                           tracks['start_y'], tracks['end_y'],
                                           tracks['start_z'], tracks['end_z'],
                                             norm_t, tracks['shower']):
        if (current_shower!=shower):
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.scatter(z_list, x_list, c=cmap(t_list), s=1)
            ax.set_xlabel('z / m')
            ax.set_ylabel('x / m')
            ax.set_title(title)
            x_list = []
            y_list = []
            z_list = []
            t_list = []
            current_shower = shower
        
        x_list.append(x1)
        x_list.append(x2)
        y_list.append(y1)
        y_list.append(y2)  
        z_list.append(z1-6371000)
        z_list.append(z2-6371000)
        t_list.append(t) # assume identical
        t_list.append(t)

    
    fig = plt.figure()
    ax = fig.add_subplot(111)        
    ax.scatter(z_list, x_list, c=cmap(t_list), s=1)
    ax.set_xlabel('z / m')
    ax.set_ylabel('x / m')
    ax.set_title(title)

def C8_zy_plots(PATH, title=""):
    tracks = pd.read_parquet(PATH)
    print(f"Max energy (kinetic): {max(tracks['start_energy'])} GeV, Min energy (kinetic): {min(tracks['end_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['start_energy'])}")    
    max_t = max(tracks['end_t'])
    min_t = min(tracks['start_t'])
    norm_t = (tracks['start_t']-min_t)/max_t 
    
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    current_shower = 0
    for pdg, x1, x2, y1, y2, z1, z2, t, shower in zip(tracks['pdg'], 
                                           tracks['start_x'], tracks['end_x'],
                                           tracks['start_y'], tracks['end_y'],
                                           tracks['start_z'], tracks['end_z'],
                                             norm_t, tracks['shower']):
        if (current_shower!=shower):
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.scatter(z_list, y_list, c=cmap(t_list), s=1)
            ax.set_xlabel('z / m')
            ax.set_ylabel('y / m')
            ax.set_title(title)
            x_list = []
            y_list = []
            z_list = []
            t_list = []
            current_shower = shower
        
        x_list.append(x1)
        x_list.append(x2)
        y_list.append(y1)
        y_list.append(y2)  
        z_list.append(z1-6371000)
        z_list.append(z2-6371000)
        t_list.append(t) # assume identical
        t_list.append(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)        
    ax.scatter(z_list, y_list, c=cmap(t_list), s=1)
    ax.set_xlabel('z / m')
    ax.set_ylabel('y / m')
    ax.set_title(title)

def C8_xy_plots(PATH, title=""):
    tracks = pd.read_parquet(PATH)
    print(f"Max energy (kinetic): {max(tracks['start_energy'])} GeV, Min energy (kinetic): {min(tracks['end_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['start_energy'])}")    
    max_t = max(tracks['end_t'])
    min_t = min(tracks['start_t'])
    norm_t = (tracks['start_t']-min_t)/max_t 
    
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    current_shower = 0
    for pdg, x1, x2, y1, y2, z1, z2, t, shower in zip(tracks['pdg'], 
                                           tracks['start_x'], tracks['end_x'],
                                           tracks['start_y'], tracks['end_y'],
                                           tracks['start_z'], tracks['end_z'],
                                             norm_t, tracks['shower']):
        if (current_shower!=shower):
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.scatter(x_list, y_list, c=cmap(t_list), s=1)
            ax.set_xlabel('x / m')
            ax.set_ylabel('y / m')
            ax.set_title(title)
            x_list = []
            y_list = []
            z_list = []
            t_list = []
            current_shower = shower
        
        x_list.append(x1)
        x_list.append(x2)
        y_list.append(y1)
        y_list.append(y2)  
        z_list.append(z1-6371000)
        z_list.append(z2-6371000)
        t_list.append(t) # assume identical
        t_list.append(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)        
    ax.scatter(x_list, y_list, c=cmap(t_list), s=1)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title(title) 
        
def C8_tz_plots(PATH, title=""):
    tracks = pd.read_parquet(PATH)
    print(f"Max energy (kinetic): {max(tracks['start_energy'])} GeV, Min energy (kinetic): {min(tracks['end_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['start_energy'])}")
    max_t = max(tracks['end_t'])
    min_t = min(tracks['start_t'])
    norm_t = (tracks['start_t']-min_t)/max_t 
    
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    current_shower = 0
    #fig = plt.figure()
    for pdg, x1, x2, y1, y2, z1, z2, t, shower in zip(tracks['pdg'], 
                                           tracks['start_x'], tracks['end_x'],
                                           tracks['start_y'], tracks['end_y'],
                                           tracks['start_z'], tracks['end_z'],
                                             norm_t, tracks['shower']):
        if (current_shower!=shower):
            plt.plot(t_list, z_list, label=title)
            plt.legend()
            z_list = []
            t_list = []
            current_shower = shower
        z_list.append(z1-6371000)
        z_list.append(z2-6371000)
        t_list.append(t) # assume identical
        t_list.append(t)
    plt.xlabel('t (normalized)')
    plt.ylabel('z / m')
    plt.plot(t_list, z_list, label=title)
    plt.legend()

            
    
### CORSIKA 7

class CorsikaTrack():

    def __init__(self, path):
        self.tracks = None
        if os.path.isfile(path):
            self.tracks = path
            print('Tracks found in:', self.tracks)

    def parse_tracks(self):
        if self.tracks is None:            
            raise IOError('No track file')

        dt_track = np.dtype([('shower', np.ulonglong),
                    ('track_counter',np.ulonglong),
                    ('interaction_counter',np.ulonglong),
                    ('cerenkov_counter',np.ulonglong),
                    ('cherenk_particle_counter',np.ulonglong),                 
                    ('in_x',np.float64),
                    ('in_y',np.float64),
                    ('in_z',np.float64),
                    ('in_time',np.float64),
                    ('in_energy',np.float64),
                    ('in_weight',np.float64),
                    ('in_particleId',np.int32),
                    ('in_depth',np.float64),
                    ('out_x',np.float64),
                    ('out_y',np.float64),
                    ('out_z',np.float64),
                    ('out_time',np.float64),
                    ('out_energy',np.float64),
                    ('out_weight',np.float64),
                    ('out_particleId',np.int32),
                    ('out_depth',np.float64),])

        if os.stat(self.tracks).st_size == 0:
            print(self.tracks + ' is empty')
            return np.array([], dtype=dt_track)
       
        self.track_data = np.memmap(self.tracks, dtype=dt_track, mode='r')

        return self.track_data


def C7_3d_plots(PATH, title=""):
    track_obj = CorsikaTrack(PATH)
    tracks = track_obj.parse_tracks()
    print(f"Max energy (total): {max(tracks['in_energy'])} GeV, Min energy (total): {min(tracks['out_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['in_energy'])}")

    max_t = max(tracks['out_time'])
    min_t = min(tracks['in_time'])
    norm_t = (tracks['in_time']-min_t)/max_t 
    
    #conversion: cm to m
    conversion = 100
    
    for pdg, x1, x2, y1, y2, z1, z2, t, E_i in zip(tracks['in_particleId'], 
                                           tracks['in_x'], tracks['out_x'],
                                           tracks['in_y'], tracks['out_y'],
                                           tracks['in_z'], tracks['out_z'],
                                             norm_t, tracks['in_energy']):
        if(E_i == tracks['in_energy'][0]):
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.set_title(title)
            ax.set_xlabel('x / m')
            ax.set_ylabel('y / m')
            ax.set_zlabel('z / m')
        ax.plot([x1/conversion, x2/conversion], [y1/conversion, y2/conversion], [z1/conversion, z2/conversion], c=cm.hot(np.abs(t)))


def C7_zx_plots(PATH, title=""):
    track_obj = CorsikaTrack(PATH)
    tracks = track_obj.parse_tracks()
    print(f"Max energy (total): {max(tracks['in_energy'])} GeV, Min energy (total): {min(tracks['out_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['in_energy'])}")
    
    max_t = max(tracks['out_time'])
    min_t = min(tracks['in_time'])
    norm_t = (tracks['in_time']-min_t)/max_t 
    
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    #conversion: cm to m
    conversion = 100
    
    first_shower = True
    for pdg, x1, x2, y1, y2, z1, z2, t, E_i in zip(tracks['in_particleId'], 
                                           tracks['in_x'], tracks['out_x'],
                                           tracks['in_y'], tracks['out_y'],
                                           tracks['in_z'], tracks['out_z'],
                                             norm_t, tracks['in_energy']):
        if (E_i == tracks['in_energy'][0]):
            if (first_shower):
                first_shower = False
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.scatter(z_list, x_list, c=cmap(t_list), s=1)
            ax.set_xlabel('z / m')
            ax.set_ylabel('x / m')
            ax.set_title(title)
            x_list = []
            y_list = []
            z_list = []
            t_list = []
        
        x_list.append(x1/conversion)
        x_list.append(x2/conversion)
        y_list.append(y1/conversion)
        y_list.append(y2/conversion)  
        z_list.append(z1/conversion)
        z_list.append(z2/conversion)
        t_list.append(t) # assume identical
        t_list.append(t)

def C7_zy_plots(PATH, title=""):
    track_obj = CorsikaTrack(PATH)
    tracks = track_obj.parse_tracks()
    print(f"Max energy (total): {max(tracks['in_energy'])} GeV, Min energy (total): {min(tracks['out_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['in_energy'])}")
    
    max_t = max(tracks['out_time'])
    min_t = min(tracks['in_time'])
    norm_t = (tracks['in_time']-min_t)/max_t 
    
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    #conversion: cm to m
    conversion = 100
    
    first_shower = True
    for pdg, x1, x2, y1, y2, z1, z2, t, E_i in zip(tracks['in_particleId'], 
                                           tracks['in_x'], tracks['out_x'],
                                           tracks['in_y'], tracks['out_y'],
                                           tracks['in_z'], tracks['out_z'],
                                             norm_t, tracks['in_energy']):
        if (E_i == tracks['in_energy'][0]):
            if (first_shower):
                first_shower = False
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.scatter(z_list, y_list, c=cmap(t_list), s=1)
            ax.set_xlabel('z / m')
            ax.set_ylabel('y / m')
            ax.set_title(title)
            x_list = []
            y_list = []
            z_list = []
            t_list = []
        
        x_list.append(x1/conversion)
        x_list.append(x2/conversion)
        y_list.append(y1/conversion)
        y_list.append(y2/conversion)  
        z_list.append(z1/conversion)
        z_list.append(z2/conversion)
        t_list.append(t) # assume identical
        t_list.append(t)       

def C7_xy_plots(PATH, title=""):
    track_obj = CorsikaTrack(PATH)
    tracks = track_obj.parse_tracks()
    print(f"Max energy (total): {max(tracks['in_energy'])} GeV, Min energy (total): {min(tracks['out_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['in_energy'])}")
    
    max_t = max(tracks['out_time'])
    min_t = min(tracks['in_time'])
    norm_t = (tracks['in_time']-min_t)/max_t 
    
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    
    cmap = plt.cm.get_cmap('hot')
    
    #conversion: cm to m
    conversion = 100
    
    first_shower = True
    for pdg, x1, x2, y1, y2, z1, z2, t, E_i in zip(tracks['in_particleId'], 
                                           tracks['in_x'], tracks['out_x'],
                                           tracks['in_y'], tracks['out_y'],
                                           tracks['in_z'], tracks['out_z'],
                                             norm_t, tracks['in_energy']):
        if (E_i == tracks['in_energy'][0]):
            if (first_shower):
                first_shower = False
                continue
            fig = plt.figure()
            ax = fig.add_subplot(111)        
            ax.scatter(x_list, y_list, c=cmap(t_list), s=1)
            ax.set_xlabel('x / m')
            ax.set_ylabel('y / m')
            ax.set_title(title)
            x_list = []
            y_list = []
            z_list = []
            t_list = []
        
        x_list.append(x1/conversion)
        x_list.append(x2/conversion)
        y_list.append(y1/conversion)
        y_list.append(y2/conversion)  
        z_list.append(z1/conversion)
        z_list.append(z2/conversion)
        t_list.append(t) # assume identical
        t_list.append(t)        

def C7_tz_plots(PATH, title=""):
    
    track_obj = CorsikaTrack(PATH)
    tracks = track_obj.parse_tracks()
    print(f"Max energy (total): {max(tracks['in_energy'])} GeV, Min energy (total): {min(tracks['out_energy'])} GeV")
    print(f"Overall number of steps: {len(tracks['in_energy'])}")

    max_t = max(tracks['out_time'])
    min_t = min(tracks['in_time'])
    norm_t = (tracks['in_time']-min_t)/max_t 
    
    z_list = []
    t_list = []
    
    #conversion: cm to m
    conversion = 100
    
    cmap = plt.cm.get_cmap('hot')
    
    fig = plt.figure()
    for pdg, x1, x2, y1, y2, z1, z2, t, E_i in zip(tracks['in_particleId'], 
                                           tracks['in_x'], tracks['out_x'],
                                           tracks['in_y'], tracks['out_y'],
                                           tracks['in_z'], tracks['out_z'],
                                             norm_t, tracks['in_energy']):
        if (E_i == tracks['in_energy'][0]):
            plt.plot(t_list, z_list)
            z_list = []
            t_list = []
        
        z_list.append(z1/conversion)
        z_list.append(z2/conversion)
        t_list.append(t) # assume identical
        t_list.append(t)
    
    plt.xlabel('t (normalized)')
    plt.ylabel('z / m')
    plt.title(title)        