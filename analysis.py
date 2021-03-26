import pandas as pd
import math
import numpy as np
import os
from collections import Counter
from read_suspension_data import SuspensionData
from read_suspension_records import SuspensionRecords
from read_census_data import CensusData
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json
from datetime import datetime


def counts_to_percent(counter):
    """
    convert the counts in counter to percents
    """
    count_sum = sum(counter.values())
    return {key: round((val/count_sum)*100, 1) for key, val in zip(counter.keys(), counter.values())}


def hist_to_percent(vals, bin_edges):
    """
    convert a histogram to percent
    """
    return {
        edge: round(val/sum(vals), 1) for val, edge in zip(vals, bin_edges)
    }


def get_num_zipcodes(census_data):
    """
    return the total number of zipcodes for which we have demographics data
    """
    return census_data['B03002_Race'].shape[0]


def race_in_zipcode(zipcode, race, census_data):
    """
    get the percent of a given race in a given zipcode
    """
    try:
        return census_data['B03002_Race'].loc[zipcode, race]
    except KeyError:
        return 0


def race_in_top_xpct_suspension_zipcodes(suspensions, census_data, suspension_reason, percent):
    """
    percent of zip codes in top x% of the given suspension type
    that are majority black, white, and hispanic or latino
    """
    top_quartile = suspensions.loc[:, suspension_reason].quantile(percent)
    above_top_quartile = suspensions[suspensions.loc[:, suspension_reason] >= top_quartile]

    top_quartile_majority_races = [majority_race_in_zipcode(zipcode, census_data) for zipcode in above_top_quartile.index]
    top_quartile_majority_races_count = Counter(top_quartile_majority_races)

    return counts_to_percent(top_quartile_majority_races_count), above_top_quartile.shape[0]


def race_in_top_xpct_suspensions_perCapita_zipcodes(suspensions, census_data, suspension_reason, percent):
    """
    percent of zip codes in top x% of suspensions per capita
    that are majority black, white, and hispanic or latino
    """
    # convert number of suspensions to suspensions per capita
    # the race table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspensions = suspensions.loc[
        suspensions.index.intersection(census_data['B03002_Race'].index)]

    suspensions_per_capita = suspensions.loc[:, suspension_reason] / suspensions.number_of_drivers
    suspensions_per_capita = suspensions_per_capita.round(4)

    top_quartile = suspensions_per_capita.quantile(percent)
    above_top_quartile = suspensions_per_capita[suspensions_per_capita >= top_quartile]

    top_quartile_majority_races = [majority_race_in_zipcode(zipcode, census_data) for zipcode in above_top_quartile.index]
    top_quartile_majority_races_count = Counter(top_quartile_majority_races)

    return counts_to_percent(top_quartile_majority_races_count), above_top_quartile.shape[0]


def majority_race_in_zipcode(zipcode, census_data):
    """
    given a zip code determine whether the majority of ppl are black, white, or hispanic/latino

    returns a string 'white', 'black', 'hispanic_or_latino', or 'no_race_data'
    """
    try:
        row = census_data['B03002_Race'].loc[zipcode, ['percent_black', 'percent_white', 'percent_hispanic_or_latino']]
    except KeyError:
        return 'no_race_data'

    column_name = row[row == row.max()].index[0]
    return column_name.replace('percent_', '')


def majority_race_in_all_zipcodes(suspensions, census_data):
    """
    percent of zip codes that are majority black, white, and hispanic or latino
    """
    all_zipcodes = census_data['B03002_Race'].index

    all_zipcodes_majority_races = [majority_race_in_zipcode(zipcode, census_data) for zipcode in all_zipcodes]
    all_zipcodes_majority_races_count = Counter(all_zipcodes_majority_races)

    return counts_to_percent(all_zipcodes_majority_races_count)


def suspensions_by_majority_race(suspensions, census_data, suspension_reason):
    """
    how many suspensions are there among majority black, white, and hispanic or latino zip codes
    """
    # get zip codes with majority for each race
    races = ['white', 'black', 'hispanic_or_latino']
    zipcodes_by_majority_race = {race: [] for race in races}
    for zipcode in census_data['B03002_Race'].index:
        zipcodes_by_majority_race[majority_race_in_zipcode(zipcode, census_data)].append(zipcode)

    # get suspensions in each of these
    suspensions_by_majority_race = {race: [] for race in races}
    for race in races:
        for zipcode in zipcodes_by_majority_race[race]:
            try:
                suspensions_in_zipcode = suspensions.loc[zipcode, suspension_reason]
            except KeyError:
                suspensions_in_zipcode = 0

            suspensions_by_majority_race[race].append(suspensions_in_zipcode)

    # some zipcodes have 0 in the estimate for total ppl but 1 suspension
    for race in races:
        suspensions_by_majority_race[race] = [
            0 if x == math.inf else x
            for x in suspensions_by_majority_race[race]]

    # get avg and std for each race
    avg_std_suspensions_by_majority_race = {race: {} for race in races}
    for race in races:
        avg_std_suspensions_by_majority_race[race]['avg'] = round(np.nanmean(suspensions_by_majority_race[race]), 1)
        avg_std_suspensions_by_majority_race[race]['std'] = round(np.nanstd(suspensions_by_majority_race[race]), 1)
        avg_std_suspensions_by_majority_race[race]['stderror'] = round(np.nanstd(suspensions_by_majority_race[race])
                                                                       / math.sqrt(len(suspensions_by_majority_race[race])), 2)
        avg_std_suspensions_by_majority_race[race]['n_zips'] = len(suspensions_by_majority_race[race])

    return avg_std_suspensions_by_majority_race


def suspensions_per_capita_by_majority_race(suspensions, census_data, suspension_reason):
    """
    how many suspensions are there per capita among majority black, white, and hispanic or latino zip codes
    """
    # get zip codes with majority for each race
    races = ['white', 'black', 'hispanic_or_latino']
    zipcodes_by_majority_race = {race: [] for race in races}
    for zipcode in census_data['B03002_Race'].index:
        zipcodes_by_majority_race[majority_race_in_zipcode(zipcode, census_data)].append(zipcode)

    # get suspensions in each of these
    suspensions_by_majority_race = {race: [] for race in races}
    suspensions_per_capita_by_majority_race = {race: [] for race in races}
    for race in races:
        for zipcode in zipcodes_by_majority_race[race]:
            try:
                suspensions_in_zipcode = suspensions.loc[zipcode, suspension_reason]
            except KeyError:
                continue

            suspensions_by_majority_race[race].append(suspensions_in_zipcode)

            # convert to per capita (num of ppl in 10,000 with suspensions)
            # TODO determine if using total from labor sheet is more accurate
            suspensions_per_capita_by_majority_race[race].append(
                (suspensions_in_zipcode / suspensions.loc[zipcode, 'number_of_drivers']) * 10**5
            )

    # some zipcodes have 0 in the estimate for total ppl but 1 suspension
    for race in races:
        suspensions_per_capita_by_majority_race[race] = [
            0 if x == math.inf else x
            for x in suspensions_per_capita_by_majority_race[race]]

    # get avg and std for each race
    avg_std_suspensions_per_capita_by_majority_race = {race: {} for race in races}
    for race in races:
        avg_std_suspensions_per_capita_by_majority_race[race]['avg'] = round(np.nanmean(suspensions_per_capita_by_majority_race[race]), 1)
        avg_std_suspensions_per_capita_by_majority_race[race]['std'] = round(np.nanstd(suspensions_per_capita_by_majority_race[race]), 1)
        avg_std_suspensions_per_capita_by_majority_race[race]['stderror'] = round(np.nanstd(suspensions_per_capita_by_majority_race[race])
                                                                                  / math.sqrt(len(suspensions_per_capita_by_majority_race[race])), 2)
        avg_std_suspensions_per_capita_by_majority_race[race]['n_zips'] = len(suspensions_per_capita_by_majority_race[race])

    return avg_std_suspensions_per_capita_by_majority_race


def plot_correlation(x, y, x_label, y_label, fig_path, fig_size, color='k',
                     fig_title=None, x_lim=None, y_lim=None, font_size=14):
    """
    Plot the two datasets as a scatter plot with a regression line.
    """
    plt.rcParams.update({'font.size': font_size})

    try:
        m, b, r, p, se = linregress(x, y)
    except AttributeError:
        m, b, r, p, se = linregress(x, y.to_list())

    fig, ax = plt.subplots(figsize=fig_size)
    plt.plot(x, y, '.', color=color)
    plt.plot(x, (m*x) + b)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if fig_title:
        plt.title(fig_title)

    if x_lim:
        ax.set_xlim(x_lim)

    if y_lim:
        ax.set_ylim(y_lim)

    plt.savefig(fig_path)
    plt.close(fig)


def race_suspension_correlation(suspensions, census_data, suspension_reason):
    """
    plot the relationship between percent of each race in a zipcode
    and the number of suspensions of the given type.

    Also compute the rank and linear correlation between each race and the given suspension type.
    """
    races = ['percent_black', 'percent_white', 'percent_hispanic_or_latino']
    suspensions = suspensions.loc[:, suspension_reason]
    # suspensions = suspensions[suspensions > 100]

    races_percent = pd.DataFrame(index=suspensions.index, columns=races)

    for zipcode in suspensions.index:
        for race in races:
            races_percent.loc[zipcode, race] = race_in_zipcode(zipcode, race, census_data)

    # sort everything by suspensions
    suspensions = suspensions.sort_values()
    x_axis = [str(i) for i in suspensions.index]

    # compute the correlations and write them to a text file
    rank_correlations = {
        race: round(suspensions.corr(races_percent.loc[suspensions.index, race],
                                     method='spearman'),
                    2)
        for race in races
    }

    linear_correlations = {
        race: round(suspensions.corr(races_percent.loc[suspensions.index, race].astype(float),
                                     method='pearson'),
                    2)
        for race in races
    }

    outdir = os.path.join(os.getcwd(), suspension_reason)

    out_txt = os.path.join(outdir, 'race_suspension_correlations.txt')
    with open(out_txt, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write(f'Correlation between the percent of each race and the number of {suspension_reason} suspensions in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(json.dumps(rank_correlations))
        f.write('\n')
        f.write('\n')
        f.write(f'Linear correlation (pearson)\n')
        f.write(json.dumps(linear_correlations))

    # plot the relationship with each race separately
    plt.rcParams.update({'font.size': 14})
    colors = ['#d95f02', '#7570b3', '#1b9e77']

    for i, race in enumerate(races):
        formatted_race = ' '.join(race.split('_')[1:])

        plot_correlation(
            x=suspensions.values,
            y=races_percent.loc[suspensions.index, race],
            x_label=f'{suspension_reason} suspensions in a given zipcode',
            y_label=f'Percent {formatted_race}',
            fig_path=os.path.join(outdir, race + '_vsSuspensions.png'),
            fig_size=(10, 10),
            color=colors[i],
            y_lim=(0, 100),
            font_size=14)


def race_suspension_per_capita_correlation(suspensions, census_data, suspension_reason):
    """
    plot the relationship between percent of each race in a zipcode
    and the number of suspensions per capita of the given type.

    Also compute the rank and linear correlation between each race and the given suspension type.
    """
    races = ['percent_black', 'percent_white', 'percent_hispanic_or_latino']

    # convert number of suspensions to suspensions per capita
    # the race table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspensions = suspensions.loc[
        suspensions.index.intersection(census_data['B03002_Race'].index)]

    suspensions_per_capita = suspensions.loc[:, suspension_reason] / suspensions.number_of_drivers
    suspensions_per_capita = suspensions_per_capita.round(4)

    # get percent race in each zip code
    races_percent = pd.DataFrame(index=suspensions_per_capita.index, columns=races)

    for zipcode in suspensions_per_capita.index:
        for race in races:
            races_percent.loc[zipcode, race] = race_in_zipcode(zipcode, race, census_data)

    # sort everything by suspensions
    suspensions_per_capita = suspensions_per_capita.sort_values()
    x_axis = [str(i) for i in suspensions_per_capita.index]

    # compute the correlations and write them to a text file
    rank_correlations = {
        race: round(suspensions_per_capita.corr(races_percent.loc[suspensions_per_capita.index, race],
                                                method='spearman'),
                    2)
        for race in races
    }
    linear_correlations = {
        race: round(suspensions_per_capita.corr(races_percent.loc[suspensions_per_capita.index, race].astype(float),
                                                method='pearson'),
                    2)
        for race in races
    }

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'race_suspensionsPerCapita_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write('Correlation between the percent of each race and the suspensions per capita in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(json.dumps(rank_correlations))
        f.write('\n')
        f.write('\n')
        f.write('Linear correlation (pearson)\n')
        f.write(json.dumps(linear_correlations))

    # plot the relationship with each race separately
    plt.rcParams.update({'font.size': 14})
    colors = ['#d95f02', '#7570b3', '#1b9e77']

    for i, race in enumerate(races):
        formatted_race = ' '.join(race.split('_')[1:])
        plot_correlation(
            x=suspensions_per_capita.values,
            y=races_percent.loc[suspensions_per_capita.index, race],
            x_label=f'{suspension_reason} suspensions per capita in a given zip code',
            y_label=f'Percent {formatted_race}',
            fig_path=os.path.join(outdir, race + '_vsSuspensionsPerCapita.png'),
            fig_size=(10, 10),
            color=colors[i],
            y_lim=(0, 100),
            font_size=14)


def records_perPerson_by_majority_race(suspensions, suspension_records, census_data, suspension_reason):
    """
    on average how many suspensions do people with suspensions have
    among majority black, white, and hispanic or latino zip codes
    """
    # get zip codes with majority for each race
    races = ['white', 'black', 'hispanic_or_latino']
    zipcodes_by_majority_race = {race: [] for race in races}
    for zipcode in census_data['B03002_Race'].index:
        zipcodes_by_majority_race[majority_race_in_zipcode(zipcode, census_data)].append(zipcode)

    # get suspensions in each of these
    suspensions_by_majority_race = {race: [] for race in races}
    records_per_person_by_majority_race = {race: [] for race in races}
    for race in races:
        for zipcode in zipcodes_by_majority_race[race]:
            try:
                suspensions_in_zipcode = suspensions.loc[zipcode, suspension_reason]
            except KeyError:
                continue

            suspensions_by_majority_race[race].append(suspensions_in_zipcode)

            # convert to per capita (num of ppl in 10,000 with suspensions)
            try:
                records_per_person_by_majority_race[race].append(
                    suspension_records.loc[zipcode, suspension_reason] / suspensions_in_zipcode
                )
            except KeyError:
                continue

    # some zipcodes have 0 in the estimate for total ppl but 1 suspension
    for race in races:
        records_per_person_by_majority_race[race] = [
            0 if x == math.inf else x
            for x in records_per_person_by_majority_race[race]]

    # get avg and std for each race
    avg_std_records_per_person_by_majority_race = {race: {} for race in races}
    for race in races:
        avg_std_records_per_person_by_majority_race[race]['avg'] = round(np.nanmean(records_per_person_by_majority_race[race]), 2)
        avg_std_records_per_person_by_majority_race[race]['std'] = round(np.nanstd(records_per_person_by_majority_race[race]), 2)
        avg_std_records_per_person_by_majority_race[race]['stderror'] = round(np.nanstd(records_per_person_by_majority_race[race])
                                                                              / math.sqrt(len(records_per_person_by_majority_race[race])), 2)
        avg_std_records_per_person_by_majority_race[race]['n_zips'] = len(records_per_person_by_majority_race[race])

    return avg_std_records_per_person_by_majority_race


def poverty_in_top_suspension_zipcodes(suspensions, census_data, suspension_reason):
    """
    Of the top quartile of zip codes with the most suspensions,
    get the percent that have poverty rates over 5, 10, 15, ..., 100%
    """
    top_quartile = suspensions.loc[:, suspension_reason].quantile(0.75)
    above_top_quartile = suspensions[suspensions.loc[:, suspension_reason] >= top_quartile]

    # the poverty table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    poverty_rates = census_data['C17002_Poverty'].loc[
        census_data['C17002_Poverty'].index.intersection(above_top_quartile.index),
        'percent_below_poverty_level']

    vals, edges = np.histogram(poverty_rates, [i for i in range(0, 100, 5)])
    percents = hist_to_percent(vals, edges)

    percents = {int(key): val for key, val in zip(percents.keys(), percents.values()) if val != 0}

    return percents


def poverty_in_top_suspension_zipcodes_to_txt(suspensions, census_data, suspension_reason):

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'poverty_in_topSuspension_zipcodes.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write('the data here are in bins so x: 0.2, y: 0.3 means 20 percent is between x and y\n')
        f.write('\n')
        f.write(f'poverty rate in top quartile of zip codes by {suspension_reason} suspensions\n')
        f.write(json.dumps(poverty_in_top_suspension_zipcodes(suspensions, census_data, suspension_reason)))


def poverty_suspension_correlation(suspensions, census_data, suspension_reason):
    """
    correlation between poverty rate and the given suspension type
    """
    # the poverty table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspensions = suspensions.loc[
        suspensions.index.intersection(census_data['C17002_Poverty'].index), suspension_reason]

    poverty_rates = census_data['C17002_Poverty'].loc[suspensions.index, 'percent_below_poverty_level']

    # sort everything by suspensions.
    suspensions = suspensions.sort_values()
    x_axis = [str(i) for i in suspensions.index]

    # compute the correlations and write them to a text file
    rank_correlation = round(suspensions.corr(poverty_rates,
                                              method='spearman'),
                             2)

    linear_correlation = round(suspensions.corr(poverty_rates,
                                                method='pearson'),
                               2)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'poverty_suspension_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write(f'Correlation between poverty rate and the number of {suspension_reason} suspensions in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(str(rank_correlation))
        f.write('\n')
        f.write('\n')
        f.write(f'Linear correlation (pearson)\n')
        f.write(str(linear_correlation))

    plot_correlation(
        x=suspensions.values,
        y=poverty_rates.values.astype(float),
        x_label=f'{suspension_reason} suspensions',
        y_label='Poverty rate',
        fig_path=os.path.join(outdir, 'poverty_vsSuspensions.png'),
        fig_size=(10, 10),
        y_lim=(0, 100),
        font_size=14,
        fig_title=suspension_reason)


def poverty_suspension_perCapita_correlation(suspensions, census_data, suspension_reason):
    """
    correlation between poverty rate and suspensions per capita
    """
    # the poverty table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspensions = suspensions.loc[
        suspensions.index.intersection(census_data['C17002_Poverty'].index)]

    suspensions_per_capita = suspensions.loc[:, suspension_reason] / suspensions.number_of_drivers
    suspensions_per_capita = suspensions_per_capita.round(4)

    poverty_rates = census_data['C17002_Poverty'].loc[suspensions_per_capita.index, 'percent_below_poverty_level']

    # sort everything by suspensions.
    suspensions_per_capita = suspensions_per_capita.sort_values()
    x_axis = [str(i) for i in suspensions_per_capita.index]

    # compute the correlations and write them to a text file
    rank_correlation = round(suspensions_per_capita.corr(poverty_rates,
                                                         method='spearman'),
                             2)

    linear_correlation = round(suspensions_per_capita.corr(poverty_rates,
                                                           method='pearson'),
                               2)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'poverty_suspensions_perCapita_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write('Correlation between poverty rate and the suspensions per capita in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(str(rank_correlation))
        f.write('\n')
        f.write('\n')
        f.write('Linear correlation (pearson)\n')
        f.write(str(linear_correlation))

    plot_correlation(
        x=suspensions_per_capita.values,
        y=poverty_rates.values.astype(float),
        x_label=f'{suspension_reason} suspensions per capita',
        y_label='Poverty rate',
        fig_path=os.path.join(outdir, 'poverty_vsSuspensions_perCapita.png'),
        fig_size=(10, 10),
        y_lim=(0, 100),
        font_size=14,
        fig_title=suspension_reason)


def poverty_suspensions_perPerson_correlation(suspensions, suspension_records, census_data, suspension_reason):
    """
    correlation between poverty rate and suspensions per person
    """
    # the poverty table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspension_records = suspension_records.loc[
        suspension_records.index.intersection(census_data['C17002_Poverty'].index)]

    poverty_rates = census_data['C17002_Poverty'].loc[suspension_records.index, 'percent_below_poverty_level']

    # sort everything by suspensions per person.
    suspensions_per_person = suspension_records.loc[poverty_rates.index, suspension_reason] / suspensions.loc[poverty_rates.index, suspension_reason]
    suspensions_per_person = suspensions_per_person.sort_values()

    # compute the correlations and write them to a text file
    rank_correlation = round(suspensions_per_person.corr(poverty_rates,
                                                         method='spearman'),
                             2)

    linear_correlation = round(suspensions_per_person.corr(poverty_rates,
                                                           method='pearson'),
                               2)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'poverty_suspensions_perPerson_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write(f'Correlation between poverty rate and the number of {suspension_reason} suspensions per person in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(str(rank_correlation))
        f.write('\n')
        f.write('\n')
        f.write(f'Linear correlation (pearson)\n')
        f.write(str(linear_correlation))

    plot_correlation(
        x=suspensions_per_person.values,
        y=poverty_rates.values.astype(float),
        x_label=f'{suspension_reason} suspensions per person',
        y_label='Poverty rate',
        fig_path=os.path.join(outdir, 'poverty_vsSuspensions_perPerson.png'),
        fig_size=(10, 10),
        y_lim=(0, 100),
        font_size=14,
        fig_title=suspension_reason)


def unemployment_in_top_suspension_zipcodes(suspensions, census_data, suspension_reason):
    """
    Of the top quartile of zip codes with the most suspensions,
    get the percent that have unemployment rates over 10, 20, ..., 100%
    """
    top_quartile = suspensions.loc[:, suspension_reason].quantile(0.75)
    above_top_quartile = suspensions[suspensions.loc[:, suspension_reason] >= top_quartile]

    # the labor table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    unemployment_rates = census_data['B23025_LaborForce'].loc[
        census_data['B23025_LaborForce'].index.intersection(above_top_quartile.index),
        'percent_unemployed']

    vals, edges = np.histogram(unemployment_rates, [i for i in range(0, 100, 5)])
    percents = hist_to_percent(vals, edges)

    percents = {int(key): val for key, val in zip(percents.keys(), percents.values()) if val != 0}

    return percents


def unemployment_in_top_suspension_zipcodes_to_txt(suspensions, census_data, suspension_reason):

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'unemployment_in_topSuspension_zipcodes.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write('the data here are in bins so x: 0.2, y: 0.3 means 20 percent is between x and y\n')
        f.write('\n')
        f.write(f'unemployment rate in top quartile of zip codes by {suspension_reason} suspensions\n')
        f.write(json.dumps(unemployment_in_top_suspension_zipcodes(suspensions, census_data, suspension_reason)))


def unemployment_suspension_correlation(suspensions, census_data, suspension_reason):
    """
    correlation between unemployment rate and suspension of the given type
    """
    # the unemployment table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspensions = suspensions.loc[
        suspensions.index.intersection(census_data['C17002_Poverty'].index), suspension_reason]

    unemployment_rates = census_data['B23025_LaborForce'].loc[suspensions.index, 'percent_unemployed']

    # sort everything by suspensions
    suspensions = suspensions.sort_values()
    x_axis = [str(i) for i in suspensions.index]

    # compute the correlations and write them to a text file
    rank_correlation = round(suspensions.corr(unemployment_rates,
                                              method='spearman'),
                             2)

    linear_correlation = round(suspensions.corr(unemployment_rates,
                                                method='pearson'),
                               2)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'unemployment_suspension_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write(f'Correlation between unemployment rate and the number of {suspension_reason} suspensions in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(str(rank_correlation))
        f.write('\n')
        f.write('\n')
        f.write(f'Linear correlation (pearson)\n')
        f.write(str(linear_correlation))

    plot_correlation(
        x=suspensions.values,
        y=unemployment_rates.values.astype(float),
        x_label=f'{suspension_reason} suspensions',
        y_label='Unemployment rate',
        fig_path=os.path.join(outdir, 'unemployment_vsSuspensions.png'),
        fig_size=(10, 10),
        y_lim=(0, 60),
        font_size=14,
        fig_title=suspension_reason)


def unemployment_suspension_perCapita_correlation(suspensions, census_data, suspension_reason):
    """
    correlation between unemployment rate and suspensions per capita
    """
    # the unemployment table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    suspensions = suspensions.loc[
        suspensions.index.intersection(census_data['B23025_LaborForce'].index)]

    suspensions_per_capita = suspensions.loc[:, suspension_reason] / suspensions.number_of_drivers
    suspensions_per_capita = suspensions_per_capita.round(4)

    unemployment_rates = census_data['B23025_LaborForce'].loc[suspensions_per_capita.index, 'percent_unemployed']

    # sort everything by suspensions.
    suspensions_per_capita = suspensions_per_capita.sort_values()
    x_axis = [str(i) for i in suspensions_per_capita.index]

    # compute the correlations and write them to a text file
    rank_correlation = round(suspensions_per_capita.corr(unemployment_rates,
                                                         method='spearman'),
                             2)

    linear_correlation = round(suspensions_per_capita.corr(unemployment_rates,
                                                           method='pearson'),
                               2)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'unemployment_suspensions_perCapita_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write('Correlation between poverty rate and suspensions per capita in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(str(rank_correlation))
        f.write('\n')
        f.write('\n')
        f.write('Linear correlation (pearson)\n')
        f.write(str(linear_correlation))

    plot_correlation(
        x=suspensions_per_capita.values,
        y=unemployment_rates.values.astype(float),
        x_label=f'{suspension_reason} suspensions per capita',
        y_label='Unemployment rate',
        fig_path=os.path.join(outdir, 'unemployment_vsSuspensions_perCapita.png'),
        fig_size=(10, 10),
        y_lim=(0, 60),
        font_size=14,
        fig_title=suspension_reason)


def unemployment_suspensions_perPerson_correlation(suspensions, suspension_records, census_data, suspension_reason):
    """
    correlation between unemployment rate and suspensions per person of the given type
    """
    suspensions = suspensions.loc[:, suspension_reason]

    # the unemployment table doesn't include all the zipcodes so need to reindex first
    # see: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    unemployment_rates = census_data['B23025_LaborForce'].loc[
        census_data['B23025_LaborForce'].index.intersection(suspension_records.index),
        'percent_unemployed']

    # sort everything by suspensions per person
    suspensions_per_person = suspension_records.loc[unemployment_rates.index, suspension_reason] / suspensions.loc[unemployment_rates.index]
    suspensions_per_person = suspensions_per_person.sort_values()

    # compute the correlations and write them to a text file
    rank_correlation = round(suspensions_per_person.corr(unemployment_rates,
                                                         method='spearman'),
                             2)

    linear_correlation = round(suspensions_per_person.corr(unemployment_rates,
                                                           method='pearson'),
                               2)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    txtfile = os.path.join(outdir, 'unemployment_suspensions_perPerson_correlations.txt')

    with open(txtfile, 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write(f'Correlation between unemployment rate and the number of {suspension_reason} suspensions per person in a given zipcode\n')
        f.write('\n')
        f.write('Rank correlation (spearman)\n')
        f.write(str(rank_correlation))
        f.write('\n')
        f.write('\n')
        f.write(f'Linear correlation (pearson)\n')
        f.write(str(linear_correlation))

    plot_correlation(
        x=suspensions_per_person.values,
        y=unemployment_rates.values.astype(float),
        x_label=f'{suspension_reason} suspensions per person',
        y_label='Unemployment rate',
        fig_path=os.path.join(outdir, 'unemployment_vsSuspensions_perPerson.png'),
        fig_size=(10, 10),
        y_lim=(0, 60),
        font_size=14,
        fig_title=suspension_reason)


def organize_zipcode_majority_race_into_df(suspensions, census_data, suspension_reason):
    """
    get the majority race in zipcodes with most suspensions, put them into a df, and save to csv
    """
    top25, top25_num_zips = race_in_top_xpct_suspension_zipcodes(suspensions, census_data, suspension_reason, 0.75)
    top5, top5_num_zips = race_in_top_xpct_suspension_zipcodes(suspensions, census_data, suspension_reason, 0.95)
    top1, top1_num_zips = race_in_top_xpct_suspension_zipcodes(suspensions, census_data, suspension_reason, 0.99)

    df = pd.DataFrame(index=top25.keys())

    for key in top25.keys():
        df.loc[key, 'top25'] = top25[key]
    for key in top5.keys():
        df.loc[key, 'top5'] = top5[key]
    for key in top1.keys():
        df.loc[key, 'top1'] = top1[key]

    df.loc['n_zip_codes', 'top25'] = top25_num_zips
    df.loc['n_zip_codes', 'top5'] = top5_num_zips
    df.loc['n_zip_codes', 'top1'] = top1_num_zips

    df.fillna(0, inplace=True)
    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'zipcode_majority_race_quantile_suspensions.csv')
    df.to_csv(outfile)


def organize_zipcode_majority_race_perCapita_into_df(suspensions, census_data, suspension_reason):
    """
    get the majority race in zipcodes with most suspensions per capita
    and put them into a df and save to csv
    """
    top25, top25_num_zips = race_in_top_xpct_suspensions_perCapita_zipcodes(suspensions, census_data, suspension_reason, 0.75)
    top5, top5_num_zips = race_in_top_xpct_suspensions_perCapita_zipcodes(suspensions, census_data, suspension_reason, 0.95)
    top1, top1_num_zips = race_in_top_xpct_suspensions_perCapita_zipcodes(suspensions, census_data, suspension_reason, 0.99)

    df = pd.DataFrame(index=top25.keys())

    for key in top25.keys():
        df.loc[key, 'top25'] = top25[key]
    for key in top5.keys():
        df.loc[key, 'top5'] = top5[key]
    for key in top1.keys():
        df.loc[key, 'top1'] = top1[key]

    df.loc['n_zip_codes', 'top25'] = top25_num_zips
    df.loc['n_zip_codes', 'top5'] = top5_num_zips
    df.loc['n_zip_codes', 'top1'] = top1_num_zips

    df.fillna(0, inplace=True)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'zipcode_majority_race_quantile_suspensions_perCapita.csv')
    df.to_csv(outfile)


def organize_suspensions_by_zipcode_majority_race_into_df(suspensions, census_data, suspension_reason):
    """
    get the avg and std suspensions in zipcodes with majority each race
    and put them into a df and save to csv
    """
    avg_std = suspensions_by_majority_race(suspensions, census_data, suspension_reason)

    df = pd.DataFrame(index=avg_std.keys(), columns=avg_std['black'].keys())

    for race in avg_std.keys():
        for measure in avg_std[race].keys():
            df.loc[race, measure] = avg_std[race][measure]

    df.fillna(0, inplace=True)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'suspensions_by_zipcodeMajorityRace.csv')
    df.to_csv(outfile)


def organize_records_per_person_by_zipcode_majority_race_into_df(suspensions, suspension_records, census_data, suspension_reason):
    """
    get the avg and std suspensions that ppl with suspensions have
    in zip codes with majority each race
    and put them into a df and save to csv
    """
    avg_std =  records_perPerson_by_majority_race(suspensions, suspension_records, census_data, suspension_reason)

    df = pd.DataFrame(index=avg_std.keys(), columns=avg_std['black'].keys())

    for race in avg_std.keys():
        for measure in avg_std[race].keys():
            df.loc[race, measure] = avg_std[race][measure]

    df.fillna(0, inplace=True)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'records_perPerson_by_zipcodeMajorityRace.csv')
    df.to_csv(outfile)


def organize_suspensions_perCapita_by_zipcode_majority_race_into_df(suspensions, census_data, suspension_reason):
    """
    get the avg and std number of people with suspension per capita
    in zipcodes with majority each race
    and put them into a df and save to csv
    """
    avg_std = suspensions_per_capita_by_majority_race(suspensions, census_data, suspension_reason)

    df = pd.DataFrame(index=avg_std.keys(), columns=avg_std['black'].keys())

    for race in avg_std.keys():
        for measure in avg_std[race].keys():
            df.loc[race, measure] = avg_std[race][measure]

    df.fillna(0, inplace=True)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'suspensionsPerCapita_by_zipcodeMajorityRace.csv')
    df.to_csv(outfile)


def organize_records_by_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason):
    """
    get the avg and std suspensions in zipcodes with majority each race
    and put them into a df and save to csv
    """
    avg_std = suspensions_by_majority_race(suspension_records, census_data, suspension_reason)

    df = pd.DataFrame(index=avg_std.keys(), columns=avg_std['black'].keys())

    for race in avg_std.keys():
        for measure in avg_std[race].keys():
            df.loc[race, measure] = avg_std[race][measure]

    df.fillna(0, inplace=True)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'records_by_zipcodeMajorityRace.csv')
    df.to_csv(outfile)


def organize_records_perCapita_by_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason):
    """
    get the avg and std number of suspension records per capita
    in zipcodes with majority each race
    and put them into a df and save to csv
    """
    avg_std = suspensions_per_capita_by_majority_race(suspension_records, census_data, suspension_reason)

    df = pd.DataFrame(index=avg_std.keys(), columns=avg_std['black'].keys())

    for race in avg_std.keys():
        for measure in avg_std[race].keys():
            df.loc[race, measure] = avg_std[race][measure]

    df.fillna(0, inplace=True)

    outdir = os.path.join(os.getcwd(), suspension_reason)
    outfile = os.path.join(outdir, 'recordsPerCapita_by_zipcodeMajorityRace.csv')
    df.to_csv(outfile)


def save_overview_figures(suspensions, census_data):
    """
    save overview figures like total number of zipcodes we have demographics data for
    and percent of zipcodes that are majority of each race
    """
    num_zipcodes = get_num_zipcodes(census_data)
    majorities = majority_race_in_all_zipcodes(suspensions, census_data)

    with open('overview_figures.txt', 'w') as f:
        f.write('File created: ' + datetime.now().strftime("%m/%d/%Y %H:%M:%S")+'\n')
        f.write('\n')
        f.write('We have demographics for ' + str(num_zipcodes) + ' zip codes.\n')
        f.write('\n')
        f.write('Below are the percent of zip codes that are majority of each of these races\n')
        f.write(json.dumps(majorities))


def main_records(suspension_records, suspension_reasons, census_data):
    """
    performs all the computations with suspension records and saves the data to CSVs and txts
    """
    for suspension_reason in suspension_reasons:

        if not os.path.isdir(suspension_reason):
            os.mkdir(suspension_reason)

        try:
            organize_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason)
            organize_zipcode_majority_race_perCapita_into_df(suspension_records, census_data, suspension_reason)
            race_suspension_correlation(suspension_records, census_data, suspension_reason)
            race_suspension_per_capita_correlation(suspension_records, census_data, suspension_reason)
            poverty_in_top_suspension_zipcodes_to_txt(suspension_records, census_data, suspension_reason)
            poverty_suspension_correlation(suspension_records, census_data, suspension_reason)
            poverty_suspension_perCapita_correlation(suspension_records, census_data, suspension_reason)
            unemployment_in_top_suspension_zipcodes_to_txt(suspension_records, census_data, suspension_reason)
            unemployment_suspension_correlation(suspension_records, census_data, suspension_reason)
            unemployment_suspension_perCapita_correlation(suspension_records, census_data, suspension_reason)

            organize_records_perCapita_by_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason)
            organize_records_by_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason)

        except KeyError:
            continue


def main(suspensions, suspension_records, suspension_reasons, census_data):
    """
    performs all the computations and saves the data to CSVs and txts
    """
    # save_overview_figures(suspensions, census_data)

    for suspension_reason in suspension_reasons:

        if not os.path.isdir(suspension_reason):
            os.mkdir(suspension_reason)

        organize_zipcode_majority_race_into_df(suspensions, census_data, suspension_reason)
        organize_zipcode_majority_race_perCapita_into_df(suspensions, census_data, suspension_reason)
        race_suspension_correlation(suspensions, census_data, suspension_reason)
        race_suspension_per_capita_correlation(suspensions, census_data, suspension_reason)
        poverty_in_top_suspension_zipcodes_to_txt(suspensions, census_data, suspension_reason)
        poverty_suspension_correlation(suspensions, census_data, suspension_reason)
        poverty_suspension_perCapita_correlation(suspensions, census_data, suspension_reason)
        unemployment_in_top_suspension_zipcodes_to_txt(suspensions, census_data, suspension_reason)
        unemployment_suspension_correlation(suspensions, census_data, suspension_reason)
        unemployment_suspension_perCapita_correlation(suspensions, census_data, suspension_reason)

        organize_suspensions_by_zipcode_majority_race_into_df(suspensions, census_data, suspension_reason)
        organize_suspensions_perCapita_by_zipcode_majority_race_into_df(suspensions, census_data, suspension_reason)
        organize_records_per_person_by_zipcode_majority_race_into_df(suspensions, suspension_records, census_data, suspension_reason)

        # we don't have suspension records for all suspension types
        try:
            poverty_suspensions_perPerson_correlation(suspensions, suspension_records, census_data, suspension_reason)
            unemployment_suspensions_perPerson_correlation(suspensions, suspension_records, census_data, suspension_reason)
            organize_records_perCapita_by_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason)
            organize_records_by_zipcode_majority_race_into_df(suspension_records, census_data, suspension_reason)
        except KeyError:
            continue


if __name__ == "__main__":
    _suspensions = SuspensionData().load_combined_suspension_data()
    _suspension_records = SuspensionRecords().load_combined_suspension_data()
    _census_data = CensusData().load_census_data()

    suspension_reasons = list(_suspensions.columns)

    if 'number_of_drivers' in suspension_reasons:
        suspension_reasons.remove('number_of_drivers')

    main(_suspensions, _suspension_records, suspension_reasons, _census_data)

    # redo for Total Records
    main_records(_suspension_records, ['total_records'], _census_data)
