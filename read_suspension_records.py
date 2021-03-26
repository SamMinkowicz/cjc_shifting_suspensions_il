import os
import pandas as pd
from pathlib import Path


class SuspensionRecords:
    current_dir = os.path.dirname(__file__)
    folder1_path = Path.home().joinpath(current_dir, 'Folder+1_Original+License+Suspension+FOIA+Data')
    folder2_path = Path.home().joinpath(current_dir, 'Folder+2_Additional+Data+Received+January+2021')
    combined_data_filename = 'combined_suspension_records_by_zipcode.csv'
    combined_data_path = Path.home().joinpath(current_dir, combined_data_filename)

    # maps suspension type to name of the file containing the raw data
    suspension_types = {
        'automated_traffic': 'Current Automated Traffic suspensions for Chicago Jobs Council.xlsx',
        'child_support_courts': 'Current Child Support suspensions reported from court for Chicago Jobs Council.xlsx',
        'child_support_dhfs': 'Current Child Support suspensions reported from DHFS for Chicago Jobs Council.xlsx',
        'failure_to_appear': 'Current Failure to Appear Suspension data for Chicago Jobs Council.xlsx',
        'failure_to_pay': 'Current Failure to Pay stops data for Chicago Jobs Council.xlsx',
        'visitation_abuse': 'Current Visitation Abuse suspensions for Chicago Jobs Council.xlsx',
    }

    def get_suspensions_by_zipcode(self, file_name, suspension_type):
        """
        get data for the given suspension type and return it in a pandas dataframe
        with zip codes as the index and suspension type as the column name
        """
        try:
            df = pd.read_excel(self.folder1_path / file_name, usecols=['ZIP_CODE'])
        except ValueError:
            df = pd.read_excel(self.folder1_path / file_name, usecols=[2])

        df = df.value_counts()
        df = df.to_frame()
        # value_counts returns a multiindex so need to extract the zip codes from it and
        # use them as the index
        df['ZIP_CODE'] = [i[0] for i in df.index]
        df = df.set_index('ZIP_CODE')
        return df.rename(columns={df.columns[0]: suspension_type})

    def add_suspension_sum_column(self):
        """
        add column with the sum of all suspension recordss in a geographical region
        """
        columns_for_total_records = list(self.suspension_types.keys())

        if 'visitation_abuse' in columns_for_total_records:
            columns_for_total_records.remove('visitation_abuse')

        self.all_suspensions['total_records'] = self.all_suspensions.loc[:, columns_for_total_records].sum(axis=1)

    def filter_out_non_illinois_zipcodes(self):
        """
        filter out all zip codes not in illinois
        illinois zip codes are in range 60002-62999
        """
        self.all_suspensions = self.all_suspensions.loc[60002:62999, :]

    def add_num_drivers(self):
        """
        add a column with the number of drivers in each zip code
        is a separate method since its in a different format
        """
        file_name = 'Counts of drivers on database (regardless of expiration status) by zip code for CJC ran 12-18-20.xlsx'
        df = pd.read_excel(self.folder2_path / file_name, index_col=0)
        df = df.rename(columns={df.columns[0]: 'number_of_drivers'})
        self.all_suspensions = self.all_suspensions.merge(df,
                                                          how='outer',
                                                          left_index=True,
                                                          right_index=True)

    def generate_table_with_all_suspension_types(self):
        """
        combine data from all suspension types and return a pandas dataframe
        with zip codes as the index and suspension types in the columns
        """
        self.all_suspensions = pd.DataFrame()

        for suspension_type in self.suspension_types.keys():
            df = self.get_suspensions_by_zipcode(self.suspension_types[suspension_type],
                                                 suspension_type)
            self.all_suspensions = self.all_suspensions.merge(df,
                                                              how='outer',
                                                              left_index=True,
                                                              right_index=True)
        self.add_suspension_sum_column()
        self.add_num_drivers()
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
    sd = SuspensionRecords()
    df = sd.load_combined_suspension_data()
    print(df.columns)
