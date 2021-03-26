import pandas as pd
import os
from read_suspension_data import SuspensionData
from read_suspension_records import SuspensionRecords
from read_census_data import CensusData
import matplotlib.pyplot as plt


class TopTen:
    suspension_data_ppl = SuspensionData().load_combined_suspension_data()
    suspension_data_records = SuspensionRecords().load_combined_suspension_data()
    census_data = CensusData().load_census_data()

    def top_10_number_of_ppl(self):
        """
        save an excel file with info on the top 10 zip codes by number of people with
        a given suspension type. Info: number of ppl with the given suspension,
        suspensions per capita of the given suspension, % of each race, % residents living in poverty,
        and % unemployment
        """
        out_file = os.path.join(os.getcwd(), r'high_level_data_for_top_ten_zip_codes_for_suspensions_by_people.xlsx')
        xl_writer = pd.ExcelWriter(out_file)

        suspension_reasons = list(self.suspension_data_ppl.columns)

        exclude = ['total_suspensions', 'number_of_drivers']
        for i in exclude:
            if i in suspension_reasons:
                suspension_reasons.remove(i)

        for suspension_reason in suspension_reasons:
            top_ten = self.suspension_data_ppl.loc[:, suspension_reason].nlargest(10)

            n_suspensions = self.suspension_data_ppl.loc[top_ten.index, suspension_reason]
            n_suspensions.rename('number_of_people_with_this_suspension', inplace=True)

            suspensions_per_capita = self.suspension_data_ppl.loc[top_ten.index, suspension_reason] / self.suspension_data_ppl.loc[top_ten.index, 'number_of_drivers']
            suspensions_per_capita = round(suspensions_per_capita, 4)
            suspensions_per_capita.rename('suspensions_per_capita', inplace=True)

            try:
                poverty = self.census_data['C17002_Poverty'].loc[top_ten.index, 'percent_below_poverty_level']
                unemployment = self.census_data['B23025_LaborForce'].loc[top_ten.index, 'percent_unemployed']
            except KeyError:
                continue

            black = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_black']
            white = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_white']
            hispanicOrLatino = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_hispanic_or_latino']

            combined_df = pd.concat([n_suspensions, suspensions_per_capita, poverty, unemployment, black, hispanicOrLatino, white], axis=1)

            combined_df.fillna(0, inplace=True)

            combined_df.to_excel(xl_writer, sheet_name=suspension_reason)

        xl_writer.save()

    def top_10_number_of_records(self):
        """
        save an excel file with info on the top 10 zip codes by number of suspension records of
        a given suspension type. Info: number of records of the given suspension,
        suspensions per capita of the given suspension, % of each race, % residents living in poverty,
        and % unemployment
        """
        out_file = os.path.join(os.getcwd(), r'high_level_data_for_top_ten_zip_codes_for_suspensions_by_records.xlsx')
        xl_writer = pd.ExcelWriter(out_file)

        suspension_reasons = list(self.suspension_data_records.columns)

        for suspension_reason in suspension_reasons:
            top_ten = self.suspension_data_records.loc[:, suspension_reason].nlargest(10)

            n_suspensions = self.suspension_data_records.loc[top_ten.index, suspension_reason]
            n_suspensions.rename('number_of_records_with_this_suspension', inplace=True)

            suspensions_per_capita = self.suspension_data_records.loc[top_ten.index, suspension_reason] / self.suspension_data_ppl.loc[top_ten.index, 'number_of_drivers']
            suspensions_per_capita = round(suspensions_per_capita, 4)
            suspensions_per_capita.rename('suspension_records_per_capita', inplace=True)

            if suspension_reason != 'total_records':
                records_over_ppl = n_suspensions / self.suspension_data_ppl.loc[top_ten.index, suspension_reason]
                records_over_ppl.rename('suspension_records_over_number_of_people_with_suspension', inplace=True)

            try:
                poverty = self.census_data['C17002_Poverty'].loc[top_ten.index, 'percent_below_poverty_level']
                unemployment = self.census_data['B23025_LaborForce'].loc[top_ten.index, 'percent_unemployed']
            except KeyError:
                continue

            black = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_black']
            white = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_white']
            hispanicOrLatino = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_hispanic_or_latino']

            if suspension_reason != 'total_records':
                combined_df = pd.concat([n_suspensions, suspensions_per_capita, records_over_ppl,
                                        poverty, unemployment, black, hispanicOrLatino, white], axis=1)
            else:
                combined_df = pd.concat([n_suspensions, suspensions_per_capita, poverty, unemployment, black, hispanicOrLatino, white], axis=1)

            combined_df.fillna(0, inplace=True)

            combined_df.to_excel(xl_writer, sheet_name=suspension_reason)

        xl_writer.save()

    def top_10_perCapita_number_of_ppl(self):
        """
        save an excel file with info on the top 10 zip codes by people with
        a given suspension type per capita. Info: number of ppl with the given suspension,
        suspensions per capita of the given suspension, % of each race, % residents living in poverty,
        and % unemployment.

        Note that the insight from this analysis is limitec since the per capita suspensions will be really high for small counties.
        """
        out_file = os.path.join(os.getcwd(), r'high_level_data_for_top_ten_zip_codes_for_suspensions_perCapita_by_people.xlsx')
        xl_writer = pd.ExcelWriter(out_file)

        suspension_reasons = list(self.suspension_data_ppl.columns)

        exclude = ['total_suspensions', 'number_of_drivers']
        for i in exclude:
            if i in suspension_reasons:
                suspension_reasons.remove(i)

        for suspension_reason in suspension_reasons:
            suspensions_per_capita = self.suspension_data_ppl.loc[:, suspension_reason] / self.suspension_data_ppl.loc[:, 'number_of_drivers']
            suspensions_per_capita = round(suspensions_per_capita, 4)
            suspensions_per_capita.rename('suspensions_per_capita', inplace=True)

            top_ten = suspensions_per_capita.nlargest(10)

            n_suspensions = self.suspension_data_ppl.loc[top_ten.index, suspension_reason]
            n_suspensions.rename('number_of_people_with_this_suspension', inplace=True)

            try:
                poverty = self.census_data['C17002_Poverty'].loc[top_ten.index, 'percent_below_poverty_level']
            except KeyError:
                poverty = pd.Series(['no data']*10, index=top_ten.index)

            try:
                unemployment = self.census_data['B23025_LaborForce'].loc[top_ten.index, 'percent_unemployed']
            except KeyError:
                unemployment = pd.Series(['no data']*10, index=top_ten.index)

            try:
                black = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_black']
                white = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_white']
                hispanicOrLatino = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_hispanic_or_latino']
            except KeyError:
                black = pd.Series(['no data']*10, index=top_ten.index)
                white = pd.Series(['no data']*10, index=top_ten.index)
                hispanicOrLatino = pd.Series(['no data']*10, index=top_ten.index)

            combined_df = pd.concat([n_suspensions, top_ten, poverty, unemployment, black, hispanicOrLatino, white], axis=1)

            combined_df.fillna(0, inplace=True)

            combined_df.to_excel(xl_writer, sheet_name=suspension_reason)

        xl_writer.save()

    def top_10_perCapita_number_of_records(self):
        """
        save an excel file with info on the top 10 zip codes by number of suspension records of
        a given suspension type. Info: number of records of the given suspension,
        suspensions per capita of the given suspension, % of each race, % residents living in poverty,
        and % unemployment

        Note that the insight from this analysis is limitec since the per capita suspensions will be really high for small counties.
        """
        out_file = os.path.join(os.getcwd(), r'high_level_data_for_top_ten_zip_codes_for_suspensions_perCapita_by_records.xlsx')
        xl_writer = pd.ExcelWriter(out_file)

        suspension_reasons = list(self.suspension_data_records.columns)

        for suspension_reason in suspension_reasons:
            suspensions_per_capita = self.suspension_data_records.loc[:, suspension_reason] / self.suspension_data_ppl.loc[:, 'number_of_drivers']
            suspensions_per_capita = round(suspensions_per_capita, 4)
            suspensions_per_capita.rename('suspension_records_per_capita', inplace=True)

            top_ten = suspensions_per_capita.nlargest(10)

            n_suspensions = self.suspension_data_records.loc[top_ten.index, suspension_reason]
            n_suspensions.rename('number_of_records_with_this_suspension', inplace=True)

            if suspension_reason != 'total_records':
                records_over_ppl = n_suspensions / self.suspension_data_ppl.loc[top_ten.index, suspension_reason]
                records_over_ppl.rename('suspension_records_over_number_of_people_with_suspension', inplace=True)

            try:
                poverty = self.census_data['C17002_Poverty'].loc[top_ten.index, 'percent_below_poverty_level']
            except KeyError:
                poverty = pd.Series(['no data']*10, index=top_ten.index)

            try:
                unemployment = self.census_data['B23025_LaborForce'].loc[top_ten.index, 'percent_unemployed']
            except KeyError:
                unemployment = pd.Series(['no data']*10, index=top_ten.index)

            try:
                black = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_black']
                white = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_white']
                hispanicOrLatino = self.census_data['B03002_Race'].loc[top_ten.index, 'percent_hispanic_or_latino']
            except KeyError:
                black = pd.Series(['no data']*10, index=top_ten.index)
                white = pd.Series(['no data']*10, index=top_ten.index)
                hispanicOrLatino = pd.Series(['no data']*10, index=top_ten.index)

            if suspension_reason != 'total_records':
                combined_df = pd.concat([n_suspensions, top_ten, records_over_ppl,
                                        poverty, unemployment, black, hispanicOrLatino, white], axis=1)
            else:
                combined_df = pd.concat([n_suspensions, top_ten, poverty, unemployment, black, hispanicOrLatino, white], axis=1)

            combined_df.fillna(0, inplace=True)

            combined_df.to_excel(xl_writer, sheet_name=suspension_reason)

        xl_writer.save()


if __name__ == "__main__":
    c = TopTen()
    c.top_10_number_of_ppl()
    c.top_10_number_of_records()
    c.top_10_perCapita_number_of_ppl()
    c.top_10_perCapita_number_of_records()
