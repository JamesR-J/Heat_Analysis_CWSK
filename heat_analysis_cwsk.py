import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import interpolate

sns.set()

h_air = 10
T_inf = 293
T_b = 388
R1 = 0.0127
R2 = 0.0254
d = 0.0008
rho_cop = 8960
k_cop = 413
cp_cop = 389

plate_fin_w = 2 * math.pi * R1


def Q_plate_fin(h, R2, R1, d, k, w, T_b, T_inf):
    """
    straight fins adiabatic fin tip total heat transfer by fin
    h = convective coefficient
    P = surface perimeter
    k = conductive coefficient
    A_c = tip area
    theta_b = temp difference
    L = fin length
    """
    L = R2 - R1
    A_c = d * w
    P = 2 * (w + d)
    theta_b = T_b - T_inf
    bracket = math.sqrt((h * P) / (k * A_c)) * L
    return math.sqrt(h * P * k * A_c) * theta_b * math.tanh(bracket)


def fin_effectivness(w, L, d, h, k):
    """
    Fin effectivness for a straight fin, effectivness below 1 means fins should not be used
    A_f = fin area
    """
    P = 2 * (w + d)
    L_c = L + (d / 2)
    A_f = 2 * w * L_c
    A_c = w * d
    m = math.sqrt((h * P) / (k * A_c))
    return A_f / (m * L * A_c)


def fin_efficiency(h, w, d, k, L):
    """
    Fin efficiency for straight fun, must be less than 1 otherwise is not correct. Fin efficiency of 1 only occurs
    when L = 0 or Q_f = 0
    """
    A_c = w * d
    P = 2 * (w + d)
    m = math.sqrt((h * P) / (k * A_c))
    return math.tanh(m * L) / (m * L)


print(Q_plate_fin(h_air, R2, R1, d, k_cop, plate_fin_w, T_b, T_inf))
print(fin_effectivness(plate_fin_w, (R2 - R1), d, h_air, k_cop))
print(fin_efficiency(h_air, plate_fin_w, d, k_cop, (R2 - R1)))


def plate_fin_ideal_q(R2, R1, T_b, T_inf, h, d, w):
    P = 2 * (w + d)
    return h * (R2 - R1) * (T_b - T_inf) * P


def pd_stuff_fixed_r1(h_air, T_inf, T_b, R1, k_cop):
    Ap = 5.08e-6
    df = pd.read_excel(r'sweep_data_fixed_r1.xlsx')
    df2 = pd.read_excel(r'sweep_data_fixed_r1_radial.xlsx')
    df3 = pd.read_excel(r'sweep_data_fixed_r1_triangle.xlsx')

    df['Ideal_Q'] = h_air * df['1 (m^2)'] * (T_b - T_inf)
    df['Efficiency'] = df['Normal total heat flux (W)'] / df['Ideal_Q']

    df2['Plate_Fin_Q'] = df2.apply(
        lambda x: Q_plate_fin(h_air, x['R2 (m)'], R1, x['Fin root thickness (m)'], k_cop, plate_fin_w, T_b, T_inf),
        axis=1)
    df2['Plate_Fin_Ideal_Q'] = df2.apply(
        lambda x: plate_fin_ideal_q(x['R2 (m)'], R1, T_b, T_inf, h_air, x['Fin root thickness (m)'], plate_fin_w),
        axis=1)
    df2['Plate_Fin_Efficiency'] = df2['Plate_Fin_Q'] / df2['Plate_Fin_Ideal_Q']

    df2['Ideal_Q'] = h_air * df2['1 (m^2)'] * (T_b - T_inf)
    df2['Efficiency'] = df2['Normal total heat flux (W)'] / df2['Ideal_Q']

    df2['x_data_2'] = df2.apply(
        lambda x: (((x['R2 (m)'] - R1) + (x['Fin root thickness (m)'] / 2)) ** (3 / 2)) * math.sqrt(
            2 * h_air / (k_cop * (
                        ((x['R2 (m)'] - R1) + (x['Fin root thickness (m)'] / 2)) * x['Fin root thickness (m)']))),
        axis=1)

    df3['Normal total heat flux (W) new'] = df3['Normal total heat flux (W)']
    df3.loc[:1, 'Normal total heat flux (W) new'] = 0.99

    df3['Ideal_Q'] = h_air * (df3['1 (m^2)']) * (T_b - T_inf)
    df3['Efficiency'] = df3['Normal total heat flux (W)'] / df3['Ideal_Q']
    df3['Efficiency new'] = df3['Normal total heat flux (W) new'] / df3['Ideal_Q']

    window_length = 69

    d3 = savgol_filter(df3['Efficiency new'], window_length, 2)

    x_data = ((df['R2 (m)'] - R1) ** (3 / 2)) * math.sqrt(2 * h_air / (k_cop * Ap))
    x_data_2 = (((df2['R2 (m)'] - R1) + (df2['Fin root thickness (m)'] / 2)) ** (3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * Ap))
    x_data_3 = ((df3['R2 (m)'] - R1) ** (3 / 2)) * math.sqrt(2 * h_air / (k_cop * Ap))

    peaks, _ = find_peaks(df3['Efficiency'], height=0)
    troughs, _ = find_peaks(-1 * df3['Efficiency'], height=-10)

    f1 = interpolate.interp1d(x_data_3[peaks], df3['Efficiency'][peaks], fill_value='extrapolate')
    f2 = interpolate.interp1d(x_data_3[troughs], df3['Efficiency'][troughs], fill_value='extrapolate')

    plt.plot(x_data_3, d3, label='Tapered_Plate_Fin_Fixed_R1', color='lightseagreen')
    plt.fill_between(x_data_3, f1(x_data_3), f2(x_data_3), color='lightseagreen', alpha=0.2, label='Noise Limits',
                     interpolate=True)
    plt.plot(x_data_2, df2['Plate_Fin_Efficiency'], label='Straight_Plate_Fin_Fixed_R1', color='royalblue')
    # plt.plot(df2['x_data_2'], df2['Plate_Fin_Efficiency'], label='Straight_Plate_Fin_Fixed_R1_new')
    plt.plot(x_data, df['Efficiency'], label='Tapered_Radial_Fin_Fixed_R1', color='darkorchid')
    plt.plot(x_data_2, df2['Efficiency'], label='Straight_Radial_Fin_Fixed_R1', color='dimgray')
    # plt.plot(df2['x_data_2'], df2['Efficiency'], label='Straight_Radial_Fin_Fixed_R1_new')

    plt.xlabel('F_p')
    plt.ylabel('Efficiency (%)')
    plt.ylim(0, 1.1)
    plt.xlim(-0.05, 2.5)
    plt.legend()
    plt.savefig('Tapered_Plate_FixedR1_ChangingR2.png', dpi=1200)
    plt.show()

    return df


def pd_stuff_fixed_ratio(h_air, T_inf, T_b, k_cop):
    Ap = 5.08e-6
    df = pd.read_excel(r'sweep_data_fixed_ratio.xlsx')
    df2 = pd.read_excel(r'sweep_data_fixed_ratio_radial.xlsx')
    df3 = pd.read_excel(r'sweep_data_fixed_ratio_triangle.xlsx')

    df['Ideal_Q'] = h_air * df['1 (m^2)'] * (T_b - T_inf)
    df['Efficiency'] = df['Normal total heat flux (W)'] / df['Ideal_Q']

    df['Plate_Fin_Q'] = df.apply(
        lambda x: Q_plate_fin(h_air, x['R2 (m)'], x['Fin inner radius (m)'], x['Fin root thickness (1/m)'], k_cop,
                              plate_fin_w, T_b, T_inf), axis=1)
    df['Plate_Fin_Ideal_Q'] = df.apply(
        lambda x: plate_fin_ideal_q(x['R2 (m)'], x['Fin inner radius (m)'], T_b, T_inf, h_air,
                                    x['Fin root thickness (1/m)'], plate_fin_w),
        axis=1)
    df['Plate_Fin_Efficiency'] = df['Plate_Fin_Q'] / df['Plate_Fin_Ideal_Q']

    df2['Ideal_Q'] = h_air * df2['1 (m^2)'] * (T_b - T_inf)
    df2['Efficiency'] = df2['Normal total heat flux (W)'] / df2['Ideal_Q']

    df3['Normal total heat flux (W) new'] = df3['Normal total heat flux (W)']
    df3.loc[:1, 'Normal total heat flux (W) new'] = 0.645

    df3['Ideal_Q'] = h_air * (df3['1 (m^2)']) * (T_b - T_inf)
    df3['Efficiency'] = df3['Normal total heat flux (W)'] / df3['Ideal_Q']
    df3['Efficiency new'] = df3['Normal total heat flux (W) new'] / df3['Ideal_Q']

    window_length = 13

    d3 = savgol_filter(df3['Efficiency new'], window_length, 2)

    x_data = ((df['R2 (m)'] - df['Fin inner radius (m)']) ** (3 / 2)) * math.sqrt(2 * h_air / (k_cop * Ap))
    x_data_2 = (((df2['R2 (m)'] - df2['Fin inner radius (m)']) + (df2['Fin root thickness (m)'] / 2)) ** (
                3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * Ap))
    x_data_3 = ((df3['R2 (m)'] - df3['Fin inner radius (m)']) ** (3 / 2)) * math.sqrt(2 * h_air / (k_cop * Ap))

    peaks, _ = find_peaks(df3['Efficiency'], height=0)
    troughs, _ = find_peaks(-1 * df3['Efficiency'], height=-10)

    f1 = interpolate.interp1d(x_data_3[peaks], df3['Efficiency'][peaks], fill_value='extrapolate')
    f2 = interpolate.interp1d(x_data_3[troughs], df3['Efficiency'][troughs], fill_value='extrapolate')

    plt.plot(x_data_3, d3, label='Tapered_Plate_Fin_Fixed_Ratio', color='lightseagreen')
    plt.fill_between(x_data_3, f1(x_data_3), f2(x_data_3), color='lightseagreen', alpha=0.2, label='Noise Limits')
    plt.plot(x_data_2, df['Plate_Fin_Efficiency'], label='Straight_Plate_Fin_Fixed_Ratio', color='royalblue')
    plt.plot(x_data, df['Efficiency'], label='Tapered_Radial_Fin_Fixed_Ratio', color='darkorchid')
    plt.plot(x_data_2, df2['Efficiency'], label='Straight_Radial_Fin_Fixed_Ratio', color='dimgray')
    plt.xlabel('F_p')
    plt.ylabel('Efficiency (%)')
    plt.ylim(0, 1.1)
    # plt.xlim(-0.05, 5)
    plt.legend()
    plt.savefig('Tapered_Plate_FixedRatio.png', dpi=1200)
    plt.show()

    return df


def pd_stuff_q_graphs(h_air, T_inf, T_b, R1, k_cop):
    Ap = 5.08e-6
    fixed_r1_df = pd_stuff_fixed_r1(h_air, T_inf, T_b, R1, k_cop)
    fixed_ratio_df = pd_stuff_fixed_ratio(h_air, T_inf, T_b, k_cop)

    fixed_r1_x_data = ((fixed_r1_df['R2 (m)'] - R1) ** (3 / 2)) * math.sqrt(2 * h_air / (k_cop * Ap))
    fixed_ratio_x_data = ((fixed_ratio_df['R2 (m)'] - fixed_ratio_df['Fin inner radius (m)']) ** (3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * Ap))

    fixed_r1_max = fixed_r1_df.loc[fixed_r1_df['Normal total heat flux (W)'].idxmax()]

    plt.plot(fixed_r1_x_data, fixed_r1_df['Normal total heat flux (W)'], label='Fixed R1', color='lightseagreen')
    plt.plot(fixed_ratio_x_data, fixed_ratio_df['Normal total heat flux (W)'], label='Fixed R1/R2 Ratio')
    plt.axvline(((fixed_r1_max['R2 (m)'] - R1) ** (3 / 2)) * math.sqrt(2 * h_air / (k_cop * Ap)), color='r',
                linestyle='--')
    plt.axhline(fixed_r1_max['Normal total heat flux (W)'], color='r', linestyle='--')
    plt.xlabel('F_p')
    plt.ylabel('Q_f')
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 20)
    plt.savefig('Tapered_Plate_Ratios_Efficiency.png', dpi=1200)
    plt.show()

    print(str(fixed_r1_max['Normal total heat flux (W)']) + 'W Max Efficiency with d_fin = ' + str(
        fixed_r1_max['Fin root thickness (1/m)']) + ' and R2 = ' + str(fixed_r1_max['R2 (m)']) + ' and R1 = ' + str(R1))

    return


def conical_pin_volume(R, h):
    return math.pi * (R ** 2) * (h / 3)


def pd_conical_pin(h_air, k_cop):
    df = pd.read_excel(r'sweep_data_conical_pin_50.xlsx')
    df2 = pd.read_excel(r'sweep_data_conical_pin_5.xlsx')
    df3 = pd.read_excel(r'sweep_data_conical_pin_0.5.xlsx')

    df['x_data'] = df.apply(lambda x: (x['Fin height (1/m^2)'] ** (3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * (0.5 * x['fin_b (m)'] * x['Fin height (1/m^2)']))), axis=1)

    df2['x_data'] = df2.apply(lambda x: (x['Fin height (1/m^2)'] ** (3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * (0.5 * x['fin_b (m)'] * x['Fin height (1/m^2)']))), axis=1)

    df3['x_data'] = df3.apply(lambda x: (x['Fin height (1/m^2)'] ** (3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * (0.5 * x['fin_b (m)'] * x['Fin height (1/m^2)']))), axis=1)

    df_dropped = df.sort_values(by=['x_data'])
    df_dropped.loc[:44, 'Normal total heat flux (W)'] = 0
    conical_max = df_dropped.loc[df_dropped['Normal total heat flux (W)'].idxmax()]

    df_dropped2 = df2.sort_values(by=['x_data'])
    df_dropped2.loc[:50, 'Normal total heat flux (W)'] = 0
    conical_max2 = df_dropped2.loc[df_dropped2['Normal total heat flux (W)'].idxmax()]

    df_dropped3 = df3.sort_values(by=['x_data'])
    df_dropped3.loc[:60, 'Normal total heat flux (W)'] = 0
    conical_max3 = df_dropped3.loc[df_dropped3['Normal total heat flux (W)'].idxmax()]

    # plt.plot(x_data, df['Normal total heat flux (W)'])
    plt.plot(df_dropped['x_data'], df_dropped['Normal total heat flux (W)'], label='50W')
    plt.plot(df_dropped2['x_data'], df_dropped2['Normal total heat flux (W)'], label='5W')
    plt.plot(df_dropped3['x_data'], df_dropped3['Normal total heat flux (W)'], label='0.5W')
    plt.axvline((conical_max['Fin height (1/m^2)'] ** (3 / 2)) * math.sqrt(
        2 * h_air / (k_cop * (0.5 * conical_max['fin_b (m)'] * conical_max['Fin height (1/m^2)']))), color='r',
                linestyle='--')
    plt.axhline(conical_max['Normal total heat flux (W)'], color='r', linestyle='--')
    plt.xlabel('F_p')
    plt.ylabel('Q_f')
    plt.legend()
    # plt.ylim(0, 0.5)
    # plt.xlim(0,3)
    plt.savefig('Conical_Pin_FixedRatio.png', dpi=1200)
    plt.show()

    print(str(conical_max['Normal total heat flux (W)']) + 'W Max Efficiency with height = ' + str(
        conical_max['Fin height (1/m^2)']) + ' and pin radius = ' + str(conical_max['fin_b (m)']))

    return


def sweep_base_pin_array():
    df = pd.read_excel(r'sweep_data_pin_array_50.xlsx')
    df2 = pd.read_excel(r'sweep_data_pin_array_5.xlsx')
    df3 = pd.read_excel(r'sweep_data_pin_array_0.5.xlsx')

    plt.plot(df['base_height (m)'], df['T - 273 (K)'], label='50W')
    plt.plot(df2['base_height (m)'], df2['T - 273 (K)'], label='5W')
    plt.plot(df3['base_height (m)'], df3['T - 273 (K)'], label='0.5W')
    plt.xlabel('Base Height (m)')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.savefig('Swept_Pin_Array_Base.png', dpi=1200)
    plt.show()

    return


def newtons(width, depth, T_inf, T_b, Q):
    return Q / ((width * depth) * (T_b - T_inf))


def initial_h():
    chip_width = 0.005
    chip_depth = 0.01
    spreader_width = 0.05
    spreader_depth = 0.025
    Q = [0.5, 5, 50]
    for i in Q:
        print('Heat transffered for chip = ' + str(i) + ' with heat transfer coefficient = ' + str(
            newtons(chip_width, chip_depth, (20 + 273), (70 + 273), i)))
        print('Heat transffered for spreader = ' + str(i) + ' with heat transfer coefficient = ' + str(
            newtons(spreader_width, spreader_depth, (20 + 273), (70 + 273), i)))

    return


def b6():
    df = pd.read_excel(r'pin_array_b6.xlsx')
    plt.plot(df['Time (s)'], df['T - 273 (K)'])
    plt.xlabel('Base Height (m)')
    plt.ylabel('Temperature (C)')
    plt.savefig('B6.png', dpi=1200)
    plt.show()

    return


def plot_time():
    from matplotlib.lines import Line2D

    df = pd.read_excel(r'b7_time.xlsx')

    ax = df.plot(x="Time (s)", y="Heating", legend=False, color='mediumaquamarine', alpha=0.8)
    ax2 = ax.twinx()
    df.plot(x="Time (s)", y="Temperature (degC), Domain Probe 2", legend=False, color='gray', ax=ax2, style='--')
    df.plot(x="Time (s)", y="Temperature (degC), Domain Probe 1", legend=False, color='black', ax=ax2)

    ax2.set_ylabel('Temperature (degC)')
    ax.set_ylabel('CPU Throttle Step Function')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    ax.set_ylim(-0.06, 1.292)
    ax.grid()
    plt.savefig('B7.png', dpi=1200)
    plt.show()


print(conical_pin_volume(0.0053524 , 0.01))

pd_stuff_q_graphs(h_air, T_inf, T_b, R1, k_cop)

print(fin_effectivness())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T_inf = 343
initial_h()

pd_conical_pin(h_air, k_cop)

sweep_base_pin_array()

b6()

plot_time()



# 800
# 80
# 8
#
# pin_radius = 0.0028
# base_height = 0.004
# 11.165W
