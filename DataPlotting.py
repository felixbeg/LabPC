import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RectangleSelector
import DataLoading as dl
# import time


class LineScan():
    def __init__(self, data_set: dl.DataSet, harmonic=2, stepx=1, stepy=1):
        # load data
        self.ifg_list = data_set.ifg_list
        self.id_list = data_set.id_list
        self.meta_data = data_set.meta_data

        # calculate the number of paths, rows, columns and averages for reshaping
        num_paths = int(np.array([id['num_path'] for id in data_set.id_list]).max()+1)
        num_rows = int(np.array([id['row'] for id in data_set.id_list]).max()+1)
        num_columns = int(np.array([id['column'] for id in data_set.id_list]).max()+1)
        num_avgs = int(np.array([id['run'] for id in data_set.id_list]).max()+1)

        # meta data
        self.harmonic = harmonic
        self.depth = self.ifg_list[0]['O1'].values.size
        self.stepx = stepx
        self.stepy = stepy

        # load spectra
        if f'C{self.harmonic}' in self.ifg_list[0].columns:
            self.spectrum = np.array([ifg[f'C{self.harmonic}_FFT'].values for ifg in self.ifg_list])
        else:
            print('No balanced detection used. Calculating the spectrum from the raw data.')
            self.spectrum = np.array([ifg[f'O{self.harmonic}_FFT'].values for ifg in self.ifg_list])

        # reshape the spectrum to the correct shape
        self.spectrum = self.spectrum.reshape((num_paths, num_rows, num_columns, num_avgs, self.depth))
        self.spectrum = np.mean(self.spectrum[:, 0, :, :, :], axis=2)
        self.wavenumbers = self.ifg_list[0]['WVNUM'].values

        # plotting settings
        self.fontsize = 14
        plt.rcParams.update({
            'font.size': self.fontsize,  # Default font size
            'axes.titlesize': self.fontsize,  # Title font size
            'axes.labelsize': self.fontsize,  # Axis label font size
            'xtick.labelsize': self.fontsize,  # X-axis tick font size
            'ytick.labelsize': self.fontsize,  # Y-axis tick font size
            'legend.fontsize': self.fontsize,  # Legend font size
            'figure.dpi': 100,  # Default figure dpi
        })

    def plot(self, lolim=1100, hilim=1770, vmin=None, vmax=None, cmap='viridis', norm_region=(55, 70)):
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)  # Make space for the slider and text box

        # Initial wavenumber and reference region, CHECK THIS
        lolim_index = np.where(self.wavenumbers > lolim)[0][0]
        hilim_index = np.where(self.wavenumbers > hilim)[0][0]
        roi = np.where((self.wavenumbers > lolim) & (self.wavenumbers < hilim))
        wavenumber_init = np.where(self.wavenumbers > 1300)[0][0]

        # Define reference region and calculate the image data
        ref_spectrum = np.mean(self.spectrum[:, norm_region[0]:norm_region[1], :], axis=1)
        image_data = np.angle(self.spectrum[:, :, wavenumber_init]/ref_spectrum[:, np.newaxis, wavenumber_init])

        # Display the image
        aspect = self.stepx/self.stepy  # Aspect ratio
        img = ax.imshow(image_data, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Paths')

        # Add a color bar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Phase Angle (radians)')

        # Slider for selecting wavenumber
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Position of slider
        slider = Slider(ax_slider, 'Wavenumber', lolim_index, hilim_index, valinit=wavenumber_init, valstep=1)

        # Norm region text box
        ax_norm = plt.axes([0.2, 0.05, 0.6, 0.03])  # Position for the text box
        norm_text_box = TextBox(ax_norm, 'Norm Region', initial=f'{norm_region[0]}, {norm_region[1]}')

        # Update function for slider
        def update(val):
            wavenumber = int(slider.val)  # Get the integer wavenumber
            lo, hi = map(int, norm_text_box.text.split(','))  # Get the norm region

            ref_spectrum = np.mean(self.spectrum[:, lo:hi, :], axis=1)
            image_data = np.angle(self.spectrum[:, :, wavenumber]/ref_spectrum[:, np.newaxis, wavenumber])

            img.set_data(image_data)  # Update the image
            ax.set_title(rf'Wavenumber: {self.wavenumbers[wavenumber]:.2f} cm$^{{-1}}$, O{self.harmonic}')
            fig.canvas.draw_idle()  # Redraw the figure

        # Connect the slider to the update function
        slider.on_changed(update)

        # Update function for text box
        def update_norm_region(text):
            update(slider.val)  # Reuse the slider update function

        # Connect the text box to the update function
        norm_text_box.on_submit(update_norm_region)

        def on_click(event):
            if event.button != 3:
                return
            if event.inaxes not in [ax]:
                return
            x, y = int(event.xdata), int(event.ydata)
            print(f'Clicked on path {y}, column {x}')

            clicked_norm = np.angle(self.spectrum[y, x, :]/ref_spectrum[y, :])

            # open new figure with the spectrum
            fig_click, ax_click = plt.subplots(figsize=(8, 6))
            ax_click.plot(self.wavenumbers[roi], clicked_norm[roi])
            ax_click.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
            ax_click.set_ylabel(rf'Phase $\phi_{self.harmonic}$ (rad)')
            ax_click.set_title(rf'Nano FTIR phase of pixel ({y}, {x})')

            fig_click.show()

        def on_select(eclick, erelease):
            print('HII')
            if eclick.inaxes != ax or erelease.inaxes != ax:
                return
            x1, y1, = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            selected_norms = np.angle(
                self.spectrum[y1:y2, x1:x2, :]/ref_spectrum[y1:y2, np.newaxis, :]
                ).reshape(-1, self.depth)

            selected_norm_mean = np.mean(selected_norms, axis=0)
            psd = np.std(selected_norms, axis=0)
            mean_psd = np.mean(psd[roi])

            # open new figure with the spectrum
            fig_rec, ax_rec = plt.subplots(figsize=(8, 6))

            cmap_oranges = plt.cm.Oranges(np.linspace(0.5, 1, len(selected_norms)))
            for i, selected_norm in enumerate(selected_norms):
                ax_rec.plot(self.wavenumbers[roi], selected_norm[roi], alpha=0.9, lw=1, c=cmap_oranges[i])
            ax_rec.plot(self.wavenumbers[roi], selected_norm_mean[roi], lw=1.5,
                        label=f'Mean\nPSD: {mean_psd*1e3:.2f} mrad', c='tab:orange')

            ax_rec.legend()
            ax_rec.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
            ax_rec.set_ylabel(rf'Phase $\phi_{self.harmonic}$ (rad)')
            ax_rec.set_title(rf'Nano FTIR phase of region ({y1}, {x1}) to ({y2}, {x2})')

            fig_rec.show()

        fig.canvas.mpl_connect('button_press_event', on_click)

        self.rect = RectangleSelector(ax, on_select, useblit=True, button=[1], minspanx=4,
                                      minspany=4, spancoords='pixels', interactive=True)

        # Initial title
        ax.set_title(rf'Wavenumber: {self.wavenumbers[wavenumber_init]:.2f} cm$^{{-1}}$, O{self.harmonic}')

        fig.show()


class StandardTwoPoint():
    def __init__(self, data_sample: dl.DataSet, data_substrate: dl.DataSet, harmonic=2, no_averaging=False):
        # load data
        self.ifg_list_sample = data_sample.ifg_list
        self.id_list_sample = data_sample.id_list

        self.ifg_list_substrate = data_substrate.ifg_list
        self.id_list_substrate = data_substrate.id_list

        # calculate the number of paths, rows, columns and averages for reshaping
        self.num_paths_sample = len(set([id['num_path'] for id in data_sample.id_list]))
        self.num_rows_sample = int(np.array([id['row'] for id in data_sample.id_list]).max()+1)
        self.num_columns_sample = int(np.array([id['column'] for id in data_sample.id_list]).max()+1)
        self.num_avgs_sample = int(np.array([id['run'] for id in data_sample.id_list]).max()+1)

        self.num_paths_substrate = len(set([id['num_path'] for id in data_substrate.id_list]))
        self.num_rows_substrate = int(np.array([id['row'] for id in data_substrate.id_list]).max()+1)
        self.num_columns_substrate = int(np.array([id['column'] for id in data_substrate.id_list]).max()+1)
        self.num_avgs_substrate = int(np.array([id['run'] for id in data_substrate.id_list]).max()+1)

        # check if the data set is correct, should have no rows and columns
        # TODO MAKE THIS MORE FLEXIBLE
        if sum([self.num_rows_sample, self.num_columns_sample,
                self.num_rows_substrate, self.num_columns_substrate]) != 4:
            raise ValueError('Columns and rows are not zero. Wrong data set?')

        # load meta data
        self.meta_data = data_sample.meta_data
        self.depth = self.ifg_list_sample[0]['O1'].values.size
        self.harmonic = harmonic

        # calculate necessary quantities
        self.calc(no_averaging=no_averaging)

        # plotting settings
        self.fontsize = 14
        plt.rcParams.update({
            'font.size': self.fontsize,  # Default font size
            'axes.titlesize': self.fontsize,  # Title font size
            'axes.labelsize': self.fontsize,  # Axis label font size
            'xtick.labelsize': self.fontsize,  # X-axis tick font size
            'ytick.labelsize': self.fontsize,  # Y-axis tick font size
            'legend.fontsize': self.fontsize,  # Legend font size
            'figure.dpi': 100,  # Default figure dpi
        })

    # TODO/CHECK: NO AVERAGING DOES NOT WORK CORRECTLY!
    def calc(self, no_averaging=False):
        # load spectra for specified harmonic for sample and substrate
        if f'C{self.harmonic}' in self.ifg_list_sample[0].columns:
            self.spectrum_sample = np.array([ifg[f'C{self.harmonic}_FFT'].values for ifg in self.ifg_list_sample])
            self.spectrum_sample_raw = np.array(
                [ifg[f'O{self.harmonic}_FFT'].values for ifg in self.ifg_list_sample])
        else:
            print('No balanced detection used. Calculating the spectrum from the raw data.\n')
            self.spectrum_sample = np.array([ifg[f'O{self.harmonic}_FFT'].values for ifg in self.ifg_list_sample])

        if f'C{self.harmonic}' in self.ifg_list_substrate[0].columns:
            self.spectrum_substrate = np.array([ifg[f'C{self.harmonic}_FFT'].values for ifg in self.ifg_list_substrate])
            self.spectrum_substrate_raw = np.array(
                [ifg[f'O{self.harmonic}_FFT'].values for ifg in self.ifg_list_substrate])
        else:
            print('No balanced detection used. Calculating the spectrum from the raw data.\n')
            self.spectrum_substrate = np.array([ifg[f'O{self.harmonic}_FFT'].values for ifg in self.ifg_list_substrate])

        # reshape the spectrum to the correct shape
        self.spectrum_sample = self.spectrum_sample.reshape(
            (self.num_paths_sample, self.num_rows_sample, self.num_columns_sample,
             self.num_avgs_sample, self.depth))
        self.spectrum_substrate = self.spectrum_substrate.reshape(
            (self.num_paths_substrate, self.num_rows_substrate, self.num_columns_substrate,
             self.num_avgs_substrate, self.depth))

        if not no_averaging:
            # only average over the 'RUN' axis, remove the 'ROW' and 'COLUMN' axes (slice with 0)
            self.spectrum_sample = np.mean(self.spectrum_sample[:, 0, 0, :, :], axis=1)
            self.spectrum_substrate = np.mean(self.spectrum_substrate[:, 0, 0, :, :], axis=1)
        # TODO CHECK IF THIS WORKS CORRECTLY
        else:
            self.spectrum_sample = self.spectrum_sample.reshape(-1, self.depth)
            self.spectrum_substrate = self.spectrum_substrate.reshape(-1, self.depth)

        # load raw data for comparison
        if f'C{self.harmonic}' in self.ifg_list_sample[0].columns:
            self.spectrum_sample_raw = self.spectrum_sample_raw.reshape(
                (self.num_paths_sample, self.num_rows_sample, self.num_columns_sample,
                 self.num_avgs_sample, self.depth))
            self.spectrum_substrate_raw = self.spectrum_substrate_raw.reshape(
                (self.num_paths_substrate, self.num_rows_substrate, self.num_columns_substrate,
                 self.num_avgs_substrate, self.depth))

            if not no_averaging:
                # only average over the 'RUN' axis, remove the 'ROW' and 'COLUMN' axes (slice with 0)
                self.spectrum_sample_raw = np.mean(self.spectrum_sample_raw[:, 0, 0, :, :], axis=1)
                self.spectrum_substrate_raw = np.mean(self.spectrum_substrate_raw[:, 0, 0, :, :], axis=1)
            # TODO CHECK IF THIS WORKS CORRECTLY
            else:
                self.spectrum_sample_raw = self.spectrum_sample_raw.reshape(-1, self.depth)
                self.spectrum_substrate_raw = self.spectrum_substrate_raw.reshape(-1, self.depth)

        # load wavenumbers and check if they are the same
        self.wavenumbers = self.ifg_list_sample[0]['WVNUM'].values
        diff = np.mean((self.wavenumbers-self.ifg_list_substrate[0]['WVNUM'].values)**2)
        if diff > 1e-6:
            raise ValueError('The wavenumbers of the sample and substrate are not the same.')

        # match first dimension of substrate to sample (for interleaved measurements)
        if self.num_paths_substrate == self.num_paths_sample+1:
            print('Interleaved measurement detected.\n')
            self.spectrum_substrate_avg = (
                self.spectrum_substrate[:-1, :] + self.spectrum_substrate[1:, :])/2
            if f'C{self.harmonic}' in self.ifg_list_substrate[0].columns:
                self.spectrum_substrate_raw_avg = (
                    self.spectrum_substrate_raw[:-1, :] + self.spectrum_substrate_raw[1:, :])/2
        # in this case, no matching needed (but no interleaved measurements)
        elif self.num_paths_substrate == self.num_paths_sample:
            print('No interleaved, but \'normal\' measurement detected.\n')
            self.spectrum_substrate_avg = np.copy(self.spectrum_substrate)
            if f'C{self.harmonic}' in self.ifg_list_substrate[0].columns:
                self.spectrum_substrate_raw_avg = np.copy(self.spectrum_substrate_raw)
        else:
            raise ValueError('Number of paths in the substrate data set does not match the sample data set.')

        # calculate the referenced spectrum
        self.referenced_spectrum = self.spectrum_sample/self.spectrum_substrate_avg
        if f'C{self.harmonic}' in self.ifg_list_sample[0].columns:
            self.referenced_spectrum_raw = self.spectrum_sample_raw/self.spectrum_substrate_raw_avg

    def plot(self, lolim=1100, hilim=1770, comp_plot=True, plot_indv=True, ylim=None):
        if comp_plot and f'C{self.harmonic}' not in self.ifg_list_sample[0].columns:
            print('No balanced detection used. Cannot plot the comparison plot.\n')
            comp_plot = False

        roi = np.where((self.wavenumbers > lolim) & (self.wavenumbers < hilim))

        # calculate key parameters like phase standard deviation and mean
        self.mean_ref_spec = np.mean(self.referenced_spectrum, axis=0)
        self.psd_ref_spec = np.std(np.angle(self.referenced_spectrum), axis=0)
        self.mean_psd = np.mean(self.psd_ref_spec[roi])
        print(f'Mean phase standard deviation: {1e3*self.mean_psd:.2f} mrad')

        if comp_plot:
            self.mean_ref_spec_raw = np.mean(self.referenced_spectrum_raw, axis=0)
            self.psd_ref_spec_raw = np.std(np.angle(self.referenced_spectrum_raw), axis=0)
            self.mean_psd_raw = np.mean(self.psd_ref_spec_raw[roi])
            print(f'Mean phase standard deviation (raw): {1e3*self.mean_psd_raw:.2f} mrad')
            print(f'Improvement factor of means: {self.mean_psd_raw/self.mean_psd:.2f}')
            print(f'Mean improvement factor: {np.mean(self.psd_ref_spec_raw[roi]/self.psd_ref_spec[roi]):.2f}')

        fig, ax = plt.subplots(figsize=(8, 6))

        if comp_plot:
            if plot_indv:
                cmap_blues = plt.cm.Blues(np.linspace(0.5, 1, len(self.referenced_spectrum_raw)))
                for i, ref_spec_raw in enumerate(self.referenced_spectrum_raw):
                    ax.plot(self.wavenumbers[roi], np.angle(ref_spec_raw[roi]), color=cmap_blues[i], lw=1, alpha=0.9)
            else:
                plt.fill_between(
                    self.wavenumbers[roi], np.angle(self.mean_ref_spec_raw[roi])-self.psd_ref_spec_raw[roi],
                    np.angle(self.mean_ref_spec_raw[roi])+self.psd_ref_spec_raw[roi], color='tab:blue', alpha=0.5)

            ax.plot(self.wavenumbers[roi], np.angle(self.mean_ref_spec_raw[roi]),
                    color='tab:blue', lw=1.5, label=f'Mean PSD (raw): {1e3*self.mean_psd_raw:.2f} mrad')

        if plot_indv:
            cmap_oranges = plt.cm.Oranges(np.linspace(0.5, 1, len(self.referenced_spectrum)))
            for i, ref_spec in enumerate(self.referenced_spectrum):
                ax.plot(self.wavenumbers[roi], np.angle(ref_spec[roi]), color=cmap_oranges[i], lw=1, alpha=0.9)
        else:
            plt.fill_between(
                self.wavenumbers[roi], np.angle(self.mean_ref_spec[roi])-self.psd_ref_spec[roi],
                np.angle(self.mean_ref_spec[roi])+self.psd_ref_spec[roi], color='tab:orange', alpha=0.5)

        ax.plot(self.wavenumbers[roi], np.angle(self.mean_ref_spec[roi]),
                color='tab:orange', lw=1.5, label=f'Mean PSD: {1e3*self.mean_psd:.2f} mrad')

        ax.legend()
        ax.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
        ax.set_ylabel(rf'Phase $\phi_{self.harmonic}$ (rad)')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(rf'Nano FTIR phase of O{self.harmonic} harmonic')

        plt.show()


def quantify_data_set(data_set: dl.DataSet, lolim, hilim, harmonic=2, plot_indv=True):
    ifg_list = data_set.ifg_list
    id_list = data_set.id_list
    meta_data = data_set.meta_data
    print(meta_data)

    num_paths = len(set([id['num_path'] for id in id_list]))
    num_rows = int(np.array([id['row'] for id in id_list]).max()+1)
    num_columns = int(np.array([id['column'] for id in id_list]).max()+1)
    num_avgs = int(np.array([id['run'] for id in id_list]).max()+1)
    depth = ifg_list[0]['O1'].values.size

    window = ifg_list[0]['WDW'].values
    wavenumbers = ifg_list[0]['WVNUM'].values
    ifgs_corr = np.array([ifg[f'C{harmonic}'].values for ifg in ifg_list])
    ifgs_raw = np.array([ifg[f'O{harmonic}'].values for ifg in ifg_list])
    ifgs_aux = np.array([ifg[f'A{harmonic}'].values for ifg in ifg_list])
    specs_corr = np.array([ifg[f'C{harmonic}_FFT'].values for ifg in ifg_list])
    specs_raw = np.array([ifg[f'O{harmonic}_FFT'].values for ifg in ifg_list])

    # bring the data into the right shape
    ifgs_corr = ifgs_corr.reshape((num_paths, num_rows, num_columns, num_avgs, -1))
    ifgs_corr = np.mean(ifgs_corr[:, 0, 0, :, :], axis=1)
    ifgs_raw = ifgs_raw.reshape((num_paths, num_rows, num_columns, num_avgs, -1))
    ifgs_raw = np.mean(ifgs_raw[:, 0, 0, :, :], axis=1)
    ifgs_aux = ifgs_aux.reshape((num_paths, num_rows, num_columns, num_avgs, -1))
    ifgs_aux = np.mean(ifgs_aux[:, 0, 0, :, :], axis=1)
    specs_corr = specs_corr.reshape((num_paths, num_rows, num_columns, num_avgs, -1))
    specs_corr = np.mean(specs_corr[:, 0, 0, :, :], axis=1)
    specs_raw = specs_raw.reshape((num_paths, num_rows, num_columns, num_avgs, -1))
    specs_raw = np.mean(specs_raw[:, 0, 0, :, :], axis=1)

    matching_factors = np.array([id['matching_factor'] for id in id_list])

    # calculate quantities amp_std, self-referenced phase std and mean
    roi = np.where((wavenumbers > lolim) & (wavenumbers < hilim))
    # amplitude quantities
    amp_std_corr = np.std(np.abs(specs_corr), axis=0)
    amp_std_raw = np.std(np.abs(specs_raw), axis=0)
    # phase quantities
    sub_spec_avg_corr = (specs_corr[:-2]+specs_corr[2:])/2
    samp_spec_corr = specs_corr[1:-1]
    sub_spec_avg_raw = (specs_raw[:-2]+specs_raw[2:])/2
    samp_spec_raw = specs_raw[1:-1]

    ref_spec_corr = samp_spec_corr/sub_spec_avg_corr
    ref_spec_raw = samp_spec_raw/sub_spec_avg_raw

    phase_std_corr = np.std(np.angle(ref_spec_corr), axis=0)[roi]
    phase_std_raw = np.std(np.angle(ref_spec_raw), axis=0)[roi]
    phase_mean_corr = np.mean(np.angle(ref_spec_corr), axis=0)[roi]
    phase_mean_raw = np.mean(np.angle(ref_spec_raw), axis=0)[roi]

    mean_psd_corr = np.mean(phase_std_corr)
    mean_psd_raw = np.mean(phase_std_raw)
    improv_of_means = mean_psd_raw/mean_psd_corr

    fig, ax = plt.subplots(3, 2, figsize=(12, 12))

    # plot complex matching factors
    ax[0, 0].plot(matching_factors.real, matching_factors.imag, ls='', marker='o')
    ax[0, 0].set_xlabel('Real part')
    ax[0, 0].set_ylabel('Imaginary part')
    ax[0, 0].axhline(0, c='gray', ls='--')
    ax[0, 0].axvline(0, c='gray', ls='--')
    ax[0, 0].set_aspect('equal')
    ax[0, 0].set_title('Matching factors in complex plane.')

    # plot one interferogram with its corr, raw, aux signal and window
    ifg_idx = 0
    ax[0, 1].plot(np.real(ifgs_raw[ifg_idx]), label=rf'$I^O_{harmonic}$', c='tab:blue')
    ax[0, 1].plot(np.real(ifgs_corr[ifg_idx]), label=rf'$I^C_{harmonic}$', c='tab:orange')
    ax[0, 1].plot(np.real(ifgs_aux[ifg_idx]*np.mean(matching_factors)),
                  label=rf'$I^A_{harmonic}$ (scaled)', c='tab:green')
    ax[0, 1].plot(window*max(np.real(ifgs_corr[ifg_idx])), label='Window', c='gray')
    ax[0, 1].set_xlabel('Depth $d$ (a.u.)')
    ax[0, 1].set_ylabel(rf'Intensity $I_{harmonic}$ (a.u.)')
    ax[0, 1].set_title('Interferogram')
    ax[0, 1].legend()

    # plot the amplitude spectrum of one interferogram
    ax[1, 0].plot(wavenumbers[5:depth//2], np.abs(specs_raw[ifg_idx])[5:depth//2],
                  label=rf'$s^O_{harmonic}$', c='tab:blue')
    ax[1, 0].plot(wavenumbers[5:depth//2], np.abs(specs_corr[ifg_idx])[5:depth//2],
                  label=rf'$s^C_{harmonic}$', c='tab:orange')
    ax[1, 0].axvline(wavenumbers[roi][0], c='gray', ls='--')
    ax[1, 0].axvline(wavenumbers[roi][-1], c='gray', ls='--')
    ax[1, 0].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax[1, 0].set_ylabel(rf'Amplitude $s_{harmonic}$ (a.u.)')
    ax[1, 0].set_title('Amplitude spectrum')
    ax[1, 0].legend()

    # plot amplitude std
    ax[1, 1].plot(wavenumbers[5:depth//2], amp_std_raw[5:depth//2], label=rf'$\Delta s^O_{harmonic}$', c='tab:blue')
    ax[1, 1].plot(wavenumbers[5:depth//2], amp_std_corr[5:depth//2], label=rf'$\Delta s^C_{harmonic}$', c='tab:orange')
    ax[1, 1].axvline(wavenumbers[roi][0], c='gray', ls='--')
    ax[1, 1].axvline(wavenumbers[roi][-1], c='gray', ls='--')
    ax[1, 1].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax[1, 1].set_ylabel(rf'Amplitude std $\Delta s_{harmonic}$ (a.u.)')
    ax[1, 1].set_title('Amplitude standard deviation')
    ax[1, 1].legend()

    # plot the phase spectrum with its standard deviation around mean
    ax[2, 0].plot(wavenumbers[roi], phase_mean_raw,
                  label=rf'$\phi^O_{harmonic}$, $\Delta\phi^O_{harmonic}$ = {1e3*mean_psd_raw:.2f} mrad',
                  c='tab:blue')
    if not plot_indv:
        ax[2, 0].fill_between(wavenumbers[roi], phase_mean_raw-phase_std_raw, phase_mean_raw+phase_std_raw,
                              color='tab:blue', alpha=0.5)
    else:
        cmap_blues = plt.cm.Blues(np.linspace(0.5, 1, len(ref_spec_raw)))
        for i, ref_spec in enumerate(ref_spec_raw):
            ax[2, 0].plot(wavenumbers[roi], np.angle(ref_spec[roi]), color=cmap_blues[i], lw=1, alpha=0.9)

    ax[2, 0].plot(wavenumbers[roi], phase_mean_corr,
                  label=rf'$\phi^C_{harmonic}$, $\Delta\phi^C_{harmonic}$ = {1e3*mean_psd_corr:.2f} mrad',
                  c='tab:orange')
    if not plot_indv:
        ax[2, 0].fill_between(wavenumbers[roi], phase_mean_corr-phase_std_corr, phase_mean_corr+phase_std_corr,
                              color='tab:orange', alpha=0.5)
    else:
        cmap_oranges = plt.cm.Oranges(np.linspace(0.5, 1, len(ref_spec_corr)))
        for i, ref_spec in enumerate(ref_spec_corr):
            ax[2, 0].plot(wavenumbers[roi], np.angle(ref_spec[roi]), color=cmap_oranges[i], lw=1, alpha=0.9)

    ax[2, 0].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax[2, 0].set_ylabel(rf'Phase $\phi_{harmonic}$ (rad)')
    ax[2, 0].set_title('Phase spectrum')
    ax[2, 0].legend()

    # plot the phase standard deviation improvement
    ax[2, 1].plot(wavenumbers[roi], phase_std_raw, label=rf'$\Delta\phi^O_{harmonic}$', c='tab:blue')
    ax[2, 1].plot(wavenumbers[roi], phase_std_corr, label=rf'$\Delta\phi^C_{harmonic}$', c='tab:orange')
    ax[2, 1].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax[2, 1].set_ylabel(rf'Phase std $\Delta\phi_{harmonic}$ (rad)')
    ax[2, 1].set_title(f'Phase standard deviation improvement: {improv_of_means:.2f}')
    ax[2, 1].legend()

    mean_amp_std_raw = np.mean(amp_std_raw[roi])
    mean_amp_signal_raw = np.mean(np.abs(specs_raw)[:, roi])
    print(f'Mean amplitude standard deviation (raw): {mean_amp_std_raw:.1f}')
    print(f'Mean amplitude signal (raw): {mean_amp_signal_raw:.1f}')
    print(f'Mean ampl. SNR (raw): {mean_amp_signal_raw/mean_amp_std_raw:.1f}')

    fig.tight_layout()
    plt.show()

    # compare interleave measurements and non-interleave measurements
    non_interleaved_ref_spec = specs_corr/np.mean(specs_corr, axis=0)
    non_interleaved_ref_spec_raw = specs_raw/np.mean(specs_raw, axis=0)

    # extract times of the measurements
    hours = []
    minutes = []
    seconds = []
    for id in id_list:
        hours.append(int(id['filename'].split(' ')[-5][:2]))
        minutes.append(int(id['filename'].split(' ')[-5][2:4]))
        seconds.append(int(id['filename'].split(' ')[-5][4:6]))

    time = np.array(hours) + np.array(minutes)/60 + np.array(seconds)/3600

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    cmap_blues = plt.cm.Blues(np.linspace(0.5, 1, len(ref_spec_raw)))
    for i, ref_spec in enumerate(ref_spec_raw):
        ax[0].plot(wavenumbers[roi], np.angle(ref_spec[roi]), color=cmap_blues[i], lw=1, alpha=0.9)
    ax[0].plot(wavenumbers[roi], np.angle(np.mean(ref_spec_raw, axis=0)[roi]),
               color='k', lw=1.5)

    cmap_oranges = plt.cm.Oranges(np.linspace(0.5, 1, len(ref_spec_corr)))
    for i, ref_spec in enumerate(ref_spec_corr):
        ax[0].plot(wavenumbers[roi], np.angle(ref_spec[roi]), color=cmap_oranges[i], lw=1, alpha=0.9)
    ax[0].plot(wavenumbers[roi], np.angle(np.mean(ref_spec_corr, axis=0)[roi]),
               color='k', lw=1.5)

    ax[0].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax[0].set_ylabel(rf'Phase $\phi_{harmonic}$ (rad)')
    ax[0].set_title('Interleaved measurements\nW./w.o. BD in orange/blue')

    cmap_blues = plt.cm.Blues(np.linspace(0.5, 1, len(non_interleaved_ref_spec_raw)))
    for i, spec in enumerate(non_interleaved_ref_spec_raw):
        ax[1].plot(wavenumbers[roi], np.angle(spec[roi]), color=cmap_blues[i], lw=1, alpha=0.9)
    ax[1].plot(wavenumbers[roi], np.angle(np.mean(non_interleaved_ref_spec_raw, axis=0)[roi]),
               color='k', lw=1.5)

    cmap_oranges = plt.cm.Oranges(np.linspace(0.5, 1, len(non_interleaved_ref_spec)))
    for i, spec in enumerate(non_interleaved_ref_spec):
        ax[1].plot(wavenumbers[roi], np.angle(spec[roi]), color=cmap_oranges[i], lw=1, alpha=0.9)
    ax[1].plot(wavenumbers[roi], np.angle(np.mean(non_interleaved_ref_spec, axis=0)[roi]),
               color='k', lw=1.5)

    ax[1].set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax[1].set_ylabel(rf'Phase $\phi_{harmonic}$ (rad)')
    ax[1].set_title('Non-interleaved measurements\nW./w.o. BD in orange/blue')

    # y axis should have same range
    ylim = ax[1].get_ylim()
    ax[0].set_ylim(ylim)

    ax[2].plot(time, np.angle(np.mean(non_interleaved_ref_spec_raw[..., roi], axis=-1)), c='tab:blue')
    ax[2].plot(time, np.angle(np.mean(non_interleaved_ref_spec[..., roi], axis=-1)), c='tab:orange')

    ax[2].set_title('Phase drift from mean over time')
    ax[2].set_xlabel('Time (h)')
    ax[2].set_ylabel(rf'Phase drift $\phi_{harmonic}$ (rad)')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    if 0:
        '''Line scan scheme'''

        # fname = 'DataSetFiles/martin202501h_lineB_corr'
        # fname = 'DataSetFiles/martin202501c_lineA_corr'
        fname = 'DataSetFiles/martin202501i_lineC_corr'
        # fname = 'DataSetFiles/martin202501i_lineC_raw'
        # fname = 'DataSetFiles/myfirstls_D_2048_T_1C2_LS_corr'

        data_set = dl.load_data_set(fname)

        ls = LineScan(data_set, harmonic=2)
        ls.plot(lolim=1100, hilim=1550, vmin=-0.2, vmax=1.2, cmap='viridis', norm_region=(-10, -1))

        # parent_folder = '/Volumes/MYUSB/250130_Felix_Hyperspec 2/2025-01-31 16466 Linescans/'
        # pattern = 'D_2048_T_1C2_LS'

        # parent_folder = '/Volumes/MYUSB/martin202501c/2025-02-01 16474/'
        # pattern = 'lineA'

        # parent_folder = '/Volumes/MYUSB/martin202501h/2025-02-01 16478'
        # pattern = 'lineB'

        # parent_folder = '/Volumes/MYUSB/martin202501i/2025-02-02 16480'
        # pattern = 'lineC'

        # fname = './DataSetFiles/martinH'

        # paths = []

        # for folder in sorted(os.listdir(parent_folder)):
        #     if not folder.startswith('._') and folder.split('NF LS ')[-1] == pattern:
        #         paths.append(os.path.join(parent_folder, folder))

        # if 1:
        #     start_time = time.time()
        #     data = dl.DataSet(paths, header=30)
        #     if 1:
        #         dl.balanced_correction(data, harmonic=2, lower_bound=0.1, upper_bound=20)
        #         dl.apply_correction(data, factor='individual')
        #     dl.transform(data)
        #     dl.save_data_set(fname, data)
        #     print(f'Time elapsed: {time.time()-start_time:.2f} s')
        # else:
        #     data = dl.load_data_set(fname)

        # ls = LineScan(data, harmonic=2)
        # ls.plot(vmin=0, vmax=1, cmap='viridis')

    if 1:
        '''Standard scheme'''

        parent_folder = '/Volumes/MYUSB/Felix_Alt_BD/2025-01-11 15982/'
        pattern = 'Rng_C_1_Nts_Standard2'

        # parent_folder = '/Volumes/MYUSB/250203_Felix_LineScans/2025-02-03 16483 VaryIntTimeLongDist'
        # pattern = 'LongTime'

        # parent_folder = '/Volumes/MYUSB/SNOM_Data/250212_Felix_Cells1/2025-02-12 16601 2PointRngB/'
        # pattern = 'RngB_4096Depth_5Time_400Dist_1Nts'

        # parent_folder = '/Volumes/MYUSB/SNOM_Data/250212_Felix_Cells1/2025-02-12 16600 StandardMeas2Nts/'
        # pattern = 'Std_2048Depth_5Time_400Dist_2Nts'

        # parent_folder = '/Volumes/MYUSB/SNOM_Data/250212_Felix_Cells1/2025-02-12 16599 VaryIntTimEpoxy'
        # pattern = 'Epxy_Si_2048D_2p0_IntT_1Nts'

        # parent_folder = '/Volumes/MYUSB/SNOM_Data/250212_Felix_Cells1/2025-02-12 16596 StandardMeas/'
        # pattern = '2_Std_2048Depth_5Time_400Dist_1Nts'
        # pattern = 'Epxy_Si_4096D_5IntT_400Mu_1Nts'

        # parent_folder = '/Volumes/MYUSB/SNOM_Data/250212_Felix_Cells1/2025-02-12 16595 AuxDetFaulty'
        # pattern = 'Std_2048Depth_5Time_400Dist_1Nts'

        substrate_path = []
        sample_path = []

        i = 0
        for folder in sorted(os.listdir(parent_folder)):
            if folder.split('NF S ')[-1] == pattern:
                if i % 2 == 0:
                    substrate_path.append(os.path.join(parent_folder, folder+'/'))
                else:
                    sample_path.append(os.path.join(parent_folder, folder+'/'))
                i += 1

        data_sample = dl.DataSet(sample_path)
        print(len(sample_path))
        # dl.balanced_correction(data_sample, harmonic=2, lower_bound=0.1, upper_bound=20)
        dl.alt_balanced_correction(data_sample, harmonic=2)
        dl.apply_correction(data_sample, factor='global')
        dl.transform(data_sample)

        print(len(substrate_path))
        data_substrate = dl.DataSet(substrate_path)
        # dl.balanced_correction(data_substrate, harmonic=2, lower_bound=0.1, upper_bound=20)
        dl.alt_balanced_correction(data_substrate, harmonic=2)
        dl.apply_correction(data_substrate, factor='global')
        dl.transform(data_substrate)

        stp = StandardTwoPoint(data_sample, data_substrate, harmonic=2, no_averaging=True)
        stp.plot(lolim=1100, hilim=1770, ylim=(0, 0.8), plot_indv=True)

        # aux_ifg_a0 = data_substrate.ifg_list[0]['A0'].values
        # plt.plot(np.abs(aux_ifg_a0))
        # plt.show()
