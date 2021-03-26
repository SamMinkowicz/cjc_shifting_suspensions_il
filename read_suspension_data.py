import os
import pandas as pd
from pathlib import Path


class SuspensionData:
    current_dir = os.path.dirname(__file__)
    folder1_path = Path.home().joinpath(current_dir, 'Folder+1_Original+License+Suspension+FOIA+Data')
    folder2_path = Path.home().joinpath(current_dir, 'Folder+2_Additional+Data+Received+January+2021')
    combined_data_filename = 'combined_suspension_people_by_zipcode.csv'
    combined_data_path = Path.home().joinpath(current_dir, combined_data_filename)

    # maps suspension type to name of the file containing the raw data
    suspension_types = {
        'automated_traffic': folder1_path / 'Number of people with Automated Traffic suspension by zip code.xls',
        'child_support_courts': folder1_path / 'Number of people with Child support suspension reported from courts for Chicago Jobs Council.xlsx',
        'child_support_dhfs': folder1_path / 'Number of people with Child support suspension reported from DHFS for Chicago Jobs Council.xlsx',
        'failure_to_appear': folder1_path / 'Number of people with Failure To Appear Suspensionstop by zip code.xls',
        'failure_to_pay': folder1_path / 'Number of people with Failure To Pay stop by zip code.xls',
        'visitation_abuse': folder1_path / 'Number of people with Visitation Abuse suspension by zip code.xls',
        'safety_responsibility': folder2_path / 'Current number of persons with open TA04 suspension broken out by zip code for CJC ran 01-05-21.xlsx',
        'financial_responsibility': folder2_path / 'Current number of persons with open TA05 suspension broken out by zip code for CJC ran 01-05-21.xlsx',
        'driving_while_suspended': folder2_path / 'Current counts of persons Suspended for Driving While Suspended by zip code for CJC ran 12-21-20.xlsx',
        'number_of_drivers': folder2_path / 'Counts of drivers on database (regardless of expiration status) by zip code for CJC ran 12-18-20.xlsx'
    }

    def get_suspensions_by_zipcode(self, file_path, suspension_type):
        """
        get data for the given suspension type and return it in a pandas dataframe
        with zip codes as the index and suspension type as the column name
        """
        df = pd.read_excel(file_path, index_col=0)
        return df.rename(columns={df.columns[0]: suspension_type})

    def filter_out_non_illinois_zipcodes(self):
        """
        filter out all zip codes not in illinois
        illinois zip codes are in range 60002-62999
        """
        self.all_suspensions = self.all_suspensions.loc[60002:62999, :]

    def generate_table_with_all_suspension_types(self):
        """
        combine data from all suspension types and return a pandas dataframe
        with zip codes as the index and suspension types in the columns
        """
        self.all_suspensions = pd.DataFrame()

        for suspension_type in self.suspension_types.keys():
            df = self.get_suspensions_by_zipcode(self.suspension_types[suspension_type],
                                                 suspension_type)
            self.all_suspensions = self.all_suspensions.merge(df.iloc[:, 0],
                                                              how='outer',
                                                              left_index=True,
                                                              right_index=True)
            self.all_suspensions = self.all_suspensions.fillna(0)

        self.filter_out_non_illinois_zipcodes()

    def save_combined_suspension_data(self):
        """
        save combined suspension data to a csv
        """
        try:
            self.all_suspensions.to_csv(self.combined_data_path)
        except AttributeError:
            self.generate_table_with_all_suspension_types()
            self.all_suspensions.to_csv(self.combined_data_path)

    def load_combined_suspension_data(self):
        """
        load combined suspension data csv
        """
        try:
            df = pd.read_csv(self.combined_data_path, index_col=0)
        except FileNotFoundError:
            self.save_combined_suspension_data()
            df = pd.read_csv(self.combined_data_path, index_col=0)

        return df


if __name__ == "__main__":
    sd = SuspensionData()
    df = sd.load_combined_suspension_data()
    print(df.columns)
