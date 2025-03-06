import numpy as np
from scipy.fft import fft
import pandas as pd
import glob
import os
from scipy.optimize import minimize
import pickle


class DataSet():
    """ DataSet class to store interferograms and metadata. Interferograms are stored in form of
    a list of pandas dataframes. Metadata is stored in form of a dictionary. id_list contains
    information about the position of the interferogram in the list and the corresponding filename.
    """
    def __init__(self, paths, header=30):
        self.paths = paths
        self.ifg_list = None  # list of pandas dataframes
        self.id_list = None  # list of dictionaries containing position and filename information
        self.meta_data = None  # contains experimental info, e.g. distance of mirror, integration time, ...

        # read data
        self.read_data(header=header)

    def read_data(self, header=30):
        """Read data from the provided paths and store it in the class."""
        self.ifg_list, self.id_list, self.meta_data = read_data(self.paths, header=header)

    def show_id(self, index):
        """Prints the id (i.e. file name, row, column, ...) of the interferogram at the specified index."""
        print(self.id_list[index])

    def show_meta_data(self):
        """Shows global metadata of the data set, such as interferometer distance, tapping amplitude, ..."""
        print(self.meta_data)


def read_data(paths, header=30):
    """
    Read data from the provided paths and return a list of pandas dataframes containing the data.
    """
    # TODO define this globally
    harmonics = 6
    if not isinstance(paths, list):
        paths = [paths]

    ifg_list = []
    id_list = []

    # iterate over all paths provided
    for num_p, p in enumerate(paths):
        # check if file exists
        if not os.path.exists(p):
            raise FileNotFoundError(f'{p} does not exist')

        file_path = glob.glob(os.path.join(p, '*Interferograms.txt'))
        if not file_path:
            raise FileNotFoundError(f"No Interferograms.txt found in {p}")
        file_path = file_path[0]

        # CHECK POSITION OF LINES IN HEADER
        # Could also load this for each interferogram, would allow more flexibility but is slower
        # read metadata once for first path
        if not num_p:
            meta_data = {'date': None, 'interferometerDistance': None, 'tappingAmplitude': None,
                         'setpoint': None, 'globalMatchingFactor': None, 'baseFile': file_path}
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f'Read metadata from {file_path}')
                header_lines = f.readlines()

                meta_data['date'] = header_lines[4].split()[-2]
                meta_data['interferometerDistance'] = float(header_lines[9].split()[-1])
                meta_data['averages'] = int(header_lines[10].split()[-1])
                meta_data['integrationTime'] = float(header_lines[11].split()[-1])
                meta_data['tappingAmplitude'] = float(header_lines[19].split()[-1])
                meta_data['setpoint'] = float(header_lines[23].split()[-1])
                meta_data['globalMatchingFactor'] = None
                meta_data['second_globalMatchingFactor'] = None

        # Load data frame, calculate complex interferograms
        df = pd.read_csv(file_path, sep='\t', header=header, encoding='utf-8', dtype=np.float64)
        for h in range(harmonics):
            try:
                df[f'O{h}'] = df[f'O{h}A']*np.exp(1j*df[f'O{h}P'])
            except KeyError:
                raise KeyError('KeyError: Check header of interferogram file (usually 29, 30 or 31).')
            if 'A0A' in df.columns:
                df[f'A{h}'] = df[f'A{h}A']*np.exp(1j*df[f'A{h}P'])
            elif 'B0A' in df.columns:
                df[f'A{h}'] = df[f'B{h}A']*np.exp(1j*df[f'B{h}P'])
            else:
                print('No auxiliary data found. Proceed without balanced detection.')

        # Define column names to be stored in the dataframe
        df_columns = [f'O{i}' for i in range(harmonics)]
        if 'A0' in df.columns:
            df_columns += [f'A{i}' for i in range(harmonics)]

        # group data by row, column, run, i.e. divide into individual interferograms
        results = []
        grouped = df.groupby(['Row', 'Column', 'Run'])
        for (row, column, run), group in grouped:
            results.append(group[df_columns].copy())  # could also store whole group and not just those columns
            id_data = {'num_path': num_p, 'row': row, 'column': column, 'run': run,
                       'filename': file_path, 'matching_factor': None, 'angle': None, 'second_matching_factor': None}
            id_list.append(id_data)

        ifg_list.extend(results)

    print(f'Number of paths provided: {len(paths)}')
    print(f'Total number of interferograms read: {len(ifg_list)}\n')

    return ifg_list, id_list, meta_data


def rms(params, opt_ifg, aux_ifg, lolim, hilim):
    """RMS of noise floor. Have to define noise floor carefully. Quantity to be minimized in balanced_correction.
    """
    corr_ifg = opt_ifg-params[0]*aux_ifg*np.exp(1j*params[1])
    cor_fft = fft(corr_ifg)
    return np.sqrt(np.mean(np.abs(cor_fft[..., lolim:hilim])**2))


def balanced_correction_exp(data_set: DataSet, harmonic=2, lower_bound=0.1, upper_bound=20):
    """Calculates the scaling and phase factor for the balanced detection.
    Based on specified harmonic!
    """
    ifg_list = data_set.ifg_list
    if 'A0' not in ifg_list[0].columns:
        raise ValueError('No auxiliary data found. Proceed without balanced detection.')

    # CHECK IF MIGHT BE DONE BETTER WITHOUT REQUIRING 'PHYSCIAL' DATA LIKE MIRROR DISTANCE
    distance = data_set.meta_data['interferometerDistance']
    depth = ifg_list[0][f'O{harmonic}'].shape[0]
    frequency_resolution = 1e4/(2*distance)  # in cm^-1
    wavenumbers = np.arange(depth)*frequency_resolution

    lolim = int(4000/frequency_resolution)
    hilim = int(5000/frequency_resolution)

    if wavenumbers[-1] < 5000:
        raise ValueError('Not enough data points to cover 5000 cm^-1')

    # calculate matching factor for each interferogram stored in list
    matching_factor_list = []
    for i, ifg in enumerate(ifg_list):
        opt_ifg = ifg[f'O{harmonic}'].values
        aux_ifg = ifg[f'A{harmonic}'].values

        res = minimize(rms, [1, 0], args=(opt_ifg, aux_ifg, lolim, hilim),
                       bounds=[(lower_bound, upper_bound), (-np.pi, np.pi)])
        tmp_matching_factor = res.x[0]*np.exp(1j*res.x[1])
        matching_factor_list.append(tmp_matching_factor)

        # store matching factor in id_list for individual interferograms
        data_set.id_list[i]['matching_factor'] = tmp_matching_factor

    matching_factor_array = np.array(matching_factor_list)
    matching_factor = np.mean(matching_factor_array)

    # store global (i.e. average) matching factor in meta_data
    data_set.meta_data['globalMatchingFactor'] = matching_factor

    # for printing information
    scaling_factor = np.abs(matching_factor)
    scaling_factor_std = np.std(np.abs(matching_factor_array))
    phase_factor = np.angle(matching_factor)
    phase_factor_std = np.std(np.angle(matching_factor_array))

    print('############################')
    print('Matching factors')
    print('Scaling factor:', round(scaling_factor, 2), '+-', round(scaling_factor_std, 2))
    print('Phase factor:', round(phase_factor, 2), '+-', round(phase_factor_std, 2))  # note cyclic nature of phase
    print('############################')
    print('')


def balanced_correction_ana(data_set: DataSet, harmonic=2):
    matching_factors_list = []
    ifg_list = data_set.ifg_list
    if 'A0' not in ifg_list[0].columns:
        raise ValueError('No auxiliary data found. Proceed without balanced detection.')

    for i, ifg in enumerate(ifg_list):
        opt_ifg = ifg[f'O{harmonic}'].values
        aux_ifg = ifg[f'A{harmonic}'].values

        mf = np.sum(opt_ifg*np.conjugate(aux_ifg))/np.sum(np.abs(aux_ifg)**2)
        matching_factors_list.append(mf)

        # store matching factor in id_list for individual interferograms
        data_set.id_list[i]['matching_factor'] = mf

    matching_factors_array = np.array(matching_factors_list)
    matching_factor = np.mean(matching_factors_array)

    # store global (i.e. average) matching factor in meta_data
    data_set.meta_data['globalMatchingFactor'] = matching_factor

    # for printing information
    scaling_factor = np.abs(matching_factor)
    scaling_factor_std = np.std(np.abs(matching_factors_array))
    phase_factor = np.angle(matching_factor)

    print('############################')
    print('Matching factors')
    print('Scaling factor:', round(scaling_factor, 2), '+-', round(scaling_factor_std, 2))
    print('Phase factor:', round(phase_factor, 2))  # note cyclic nature of phase
    print('############################')
    print('')


# LC = linear combination of noise harmonics NOT IN USE
def apply_correction(data_set: DataSet, factor='global', lc=False):
    """Apply the balanced detection to the list of datda frames. Add new column 'C'
    with corrected interferograms.
    """
    ifg_list = data_set.ifg_list

    # choose global or individual matching factors
    if factor == 'global':
        matching_factor = data_set.meta_data['globalMatchingFactor']
        second_matching_factor = data_set.meta_data['second_globalMatchingFactor']
        if matching_factor is None:
            raise ValueError('No global matching factor found. Run balanced_correction first.')
        print('Applying global matching factor.\n')
    elif factor == 'individual':
        if data_set.id_list[0]['matching_factor'] is None:
            raise ValueError('No individual matching factors found. Run balanced_correction first.')
        print('Applying individual matching factors.\n')
    else:
        raise ValueError('Factor must be either global or individual.')

    if 'A0' not in ifg_list[0].columns:
        raise ValueError('No auxiliary data found. Proceed without balanced detection.')

    # here we apply the correction for each interferogram
    for i, ifg in enumerate(ifg_list):
        if factor == 'individual':
            matching_factor = data_set.id_list[i]['matching_factor']
            # idea of using a linear combination of noise harmonics
            # NOT IN USE
            if lc:
                second_matching_factor = data_set.id_list[i]['second_matching_factor']
        # note that all harmonics are corrected, but only one harmonic is used for finding the matching factors
        for h in range(6):
            ifg[f'C{h}'] = ifg[f'O{h}']-matching_factor*ifg[f'A{h}']
            if lc:
                ifg[f'C{h}'] -= second_matching_factor*ifg['A0']


def transform(data_set: DataSet):
    """Apodizes interferograms, calculates fourier transfrom and x-axis (wavenumbers).
    """
    print('Transforming interferograms.\n')
    ifg_list = data_set.ifg_list
    distance = data_set.meta_data['interferometerDistance']

    for ifg in ifg_list:
        depth = ifg['O0'].shape[0]
        frequency_resolution = 1e4/(2*distance)  # in cm^-1
        wavenumbers = np.arange(depth)*frequency_resolution
        ifg['WVNUM'] = wavenumbers
        ifg['WDW'] = window(ifg['O2'].values)  # calculate window only once, could change this

        for h in range(6):
            ifg[f'O{h}_FFT'] = fft(ifg[f'O{h}'].values*ifg['WDW'])
            if 'C0' in ifg.columns:
                ifg[f'C{h}_FFT'] = fft(ifg[f'C{h}'].values*ifg['WDW'])


def data_angle(x):
    import scipy as sp
    """
    From Theo
    Return the angle of the best fit line for the array of points (x,y), or the array x in the complex plane.
    Parameters
    ----------
    x : 1D array of complex
        x coordinates, or complexe coordinate of the points to fit.
    Returns
    -------
    angle : float
        Angle of the best fit line in radians.
    """
    xx = np.real(x)
    yy = np.imag(x)

    if np.min(xx) == np.max(xx):        # Data is perfectly vertical
        return np.pi/2
    if np.min(yy) == np.max(yy):        # Data is perfectly hozizontal
        return 0
    regressHor = sp.stats.linregress(xx, yy)
    regressVer = sp.stats.linregress(yy, xx)
    if regressHor.stderr < regressVer.stderr:
        angle = np.arctan(regressHor.slope)
    else:
        angle = np.arctan(1/regressVer.slope)
    return angle


# TODO CHECK THIS, seems to work fine though
def real_transform(data_set: DataSet):
    """Determines angle of line that interferograms lie on. Rotates that line to the real axis.
    Then takes real FFT assuming that perpendicular to the line there is only noise."""
    ifg_list = data_set.ifg_list
    distance = data_set.meta_data['interferometerDistance']

    for i, ifg in enumerate(ifg_list):
        depth = ifg['O0'].shape[0]
        frequency_resolution = 1e4/(2*distance)  # in cm^-1
        wavenumbers = np.arange(depth)*frequency_resolution
        ifg['WVNUM'] = wavenumbers
        ifg['WDW'] = window(ifg['O2'].values)  # calculate window only once, could change this

        for h in range(6):
            if 'C0' in ifg.columns:
                angle = data_angle(ifg[f'C{h}'].values)
                ifg[f'C{h}_FFT'] = fft(
                    np.real(ifg[f'C{h}'].values*np.exp(-1j*angle)*ifg['WDW'])
                    )
                if h == 2:
                    data_set.id_list[i]['angle'] = angle
            else:
                angle = data_angle(ifg[f'O{h}'].values)
                if h == 2:
                    data_set.id_list[i]['angle'] = angle
            ifg[f'O{h}_FFT'] = fft(
                np.real(ifg[f'O{h}'].values*np.exp(-1j*angle)*ifg['WDW'])
                )


def list_to_array(data_set: DataSet, channel='C', type=None):
    """Converts a list of pandas dataframes to a numpy array.
    Shape of array is (len_p, row, col, avg, harmonic, depth).
    """
    ifg_list = data_set.ifg_list
    if f'{channel}0' not in ifg_list[0].columns and f'{channel}0_FFT' not in ifg_list[0].columns:
        raise ValueError('Channel not found in interferograms.')
    print(f'Converting interferograms to array for channel {channel}.\n')

    harmonics = 6
    num_paths = int(np.array([id['num_path'] for id in data_set.id_list]).max()+1)
    num_rows = int(np.array([id['row'] for id in data_set.id_list]).max()+1)
    num_columns = int(np.array([id['column'] for id in data_set.id_list]).max()+1)
    num_avgs = int(np.array([id['run'] for id in data_set.id_list]).max()+1)
    depth = ifg_list[0][f'{channel}0'].shape[0]

    # flatten the list of dataframes
    ifg_array = np.zeros((num_paths*num_rows*num_columns*num_avgs, harmonics, depth), dtype=complex)
    for i, ifg in enumerate(ifg_list):
        for h in range(harmonics):
            if type == 'FFT':
                ifg_array[i, h] = ifg[f'{channel}{h}_FFT'].values
            else:
                ifg_array[i, h] = ifg[f'{channel}{h}'].values

    ifg_array = ifg_array.reshape(num_paths, num_rows, num_columns, num_avgs, harmonics, depth)

    return ifg_array


def window(interferogram):
    """Applies a Blackman window to the interferogram. Takes individual interferogram.
    """
    zpd = np.argmax(np.abs(interferogram))

    rightInterferogram = interferogram[zpd:]
    rightX = np.linspace(-0.5, 0.5, len(rightInterferogram))

    rightBlackman = np.blackman(len(rightX))
    leftSlope = rightBlackman[:np.argmax(rightBlackman)]
    rightSlope = rightBlackman[np.argmax(rightBlackman):]
    wndw = np.ones(len(interferogram))
    wndw[:len(leftSlope)] = leftSlope
    wndw[-len(rightSlope):] = rightSlope
    return wndw


def save_data_set(fname, data_set):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    with open(f'{fname}', 'wb') as f:
        pickle.dump(data_set, f)


def load_data_set(fname):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    with open(f'{fname}', 'rb') as f:
        data_set = pickle.load(f)
    print('Data set loaded.\n')
    return data_set


def path_finder(parent_folder: str, pattern: str):
    """Searches for folders in the parent folder that contain the specified pattern.
    Returns a list of paths to the folders. Assumes alternating substrate and sample folders.
    """
    substrate_paths = []
    sample_paths = []
    i = 0
    for folder in sorted(os.listdir(parent_folder)):
        if folder.split('NF S ')[-1] == pattern:
            if i % 2 == 0:
                substrate_paths.append(os.path.join(parent_folder, folder+'/'))
            else:
                sample_paths.append(os.path.join(parent_folder, folder+'/'))
            i += 1
    print(f'Found {len(substrate_paths)} substrate and {len(sample_paths)} sample folders.\n')
    return substrate_paths, sample_paths


def full_process(data_set: DataSet, harmonic=2, method='exp', factor='global', real_transform=False):
    if method == 'exp':
        balanced_correction_exp(data_set, harmonic=harmonic)
    elif method == 'ana':
        balanced_correction_ana(data_set, harmonic=harmonic)
    apply_correction(data_set, factor=factor)
    if real_transform:
        real_transform(data_set)
    else:
        transform(data_set)
