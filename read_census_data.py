import os
import pandas as pd
from pathlib import Path


class CensusData:
    current_dir = os.path.dirname(__file__)
    data_dir = Path.home().joinpath(current_dir, 'census_data')
    census_data_filename = 'TA_INTERNAL_HA_CJC_DATA_ACS.xlsx'
    census_data_path = data_dir / census_data_filename

    def compute_percent_each_race(self, df):
        """
        add column with percent of each race in geographical region
        """
        percent_black = round(
            (df.loc[:, 'Estimate!!Total:!!Not Hispanic or Latino:!!Black or African American alone']
            / df.loc[:, 'Estimate!!Total:']) * 100,
            1)
        df['percent_black'] = percent_black

        percent_white = round(
            (df.loc[:, 'Estimate!!Total:!!Not Hispanic or Latino:!!White alone']
            / df.loc[:, 'Estimate!!Total:']) * 100,
            1)
        df['percent_white'] = percent_white

        percent_hispanic_or_latino = round(
            (df.loc[:, 'Estimate!!Total:!!Hispanic or Latino:']
            / df.loc[:, 'Estimate!!Total:']) * 100,
            1)
        df['percent_hispanic_or_latino'] = percent_hispanic_or_latino

        return df

    def compute_percent_below_poverty_level(self, df):
        """
        add column with percent of people in geographical region
        below poverty level
        """
        percent_below = round(
            ((df.loc[:, 'Estimate!!Total:!!Under .50'] + df.loc[:, 'Estimate!!Total:!!.50 to .99'])
            / df.loc[:, 'Estimate!!Total:']) * 100,
            1)
        df['percent_below_poverty_level'] = percent_below

        return df

    def compute_percent_unemployed(self, df):
        """
        add column with percent of unemployed people in geographical region
        """
        percent_unemployed = round(
            (df.loc[:, 'Estimate!!Total:!!In labor force:!!Civilian labor force:!!Unemployed']
            / df.loc[:, 'Estimate!!Total:!!In labor force:!!Civilian labor force:']) * 100,
            1)
        df['percent_unemployed'] = percent_unemployed

        return df

    def process_census_data(self):
        """
        load and preprocess raw census data
        """
        # there is a sheet for poverty (table C17002), race (B03002), and labor force (B23025)
        # so the output will be a dict of 3 DataFrames
        dfs = pd.read_excel(self.census_data_path, sheet_name=None)

        # add percent below poverty column to the poverty df
        dfs['C17002_Poverty'] = self.compute_percent_below_poverty_level(dfs['C17002_Poverty'])

        # add percent unemplyed column to the labor force df
        dfs['B23025_LaborForce'] = self.compute_percent_unemployed(dfs['B23025_LaborForce'])

        # add percent of each race to race df
        dfs['B03002_Race'] = self.compute_percent_each_race(dfs['B03002_Race'])

        # extract zipcodes from the 'id' column and set as the indices for each df
        for sheet in dfs.keys():
            # ids are in the form '8600000USzipcode' where zipcode is a chicago zip code
            zip_codes = [int(z[-5:]) for z in dfs[sheet].loc[:, 'id']]
            dfs[sheet]['zip_code'] = zip_codes
            dfs[sheet].set_index('zip_code', inplace=True)

            dfs[sheet] = dfs[sheet].fillna(0)

        return dfs

    def load_census_data(self):
        """
        load processed census data

        there is a sheet for poverty (table C17002), race (B03002), and labor force (B23025)
        so the output will be a dict of 3 DataFrames
        """
        return self.process_census_data()


if __name__ == "__main__":
    cd = CensusData()
    data = cd.load_census_data()
    print(data.keys())
