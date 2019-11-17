import numpy as np
import pandas as pd
import re

def clean_data(path, file):
    col = pd.read_csv(path + '/' + file)
    col_dna = col.dropna(axis=1)  # drop na
    col_dtypes = dict(col_dna.dtypes.replace(np.dtype('int64'),np.dtype('float64'))) # make the dtypes floats
    col_dtypes['UNITID'] = np.dtype('int64') # convert the UNITID back to int
    vars_interest = ['ADM_RATE','UGDS','TUITIONFEE_IN','TUITIONFEE_OUT','MN_EARN_WNE_P10'] # Include these vars
    col_dtypes.update({a: np.dtype('float64') for a in vars_interest}) # make them floats

    return col_dtypes


def read_cs_data(year, col_dtypes, path):
    nextyr = str(int(year) + 1)[-2:]
    filename = path + '/MERGED{}_{}_PP.csv'.format(year,nextyr)
    col = pd.read_csv(filename,na_values='PrivacySuppressed',
                      dtype=col_dtypes,usecols=col_dtypes.keys())
    col['Year'] = pd.Period(str(int(year) + 1),freq='Y')
    return col


def tuition_data(year1, year2, col_dtypes, university):
    col = pd.concat((read_cs_data(str(y),col_dtypes, path) for y in range(year1,year2)))
    col = col.set_index(['UNITID', 'Year'])

    col_large = col[col['UGDS'] > 1000]

    city = col_large.query('STABBR=="MA" and INSTNM=="{university}"'.format(university=university))
    city = city.reset_index(level=0)

    city['YearDT'] = city.index.to_timestamp()

    city_df = city[['TUITIONFEE_OUT']]

    city_df = city_df.dropna(how='all')

    return city_df


def save_to_dir(university, file_postfix, df, path):
    university = re.sub(' ', '-', university)
    filename = university + '_' + file_postfix + '.csv'
    df.to_csv(path + filename)


if __name__ == "__main__":
    universities = ('Boston University', 'Northeastern University',
                    'Harvard University', 'Tufts University',
                    'Massachusetts Institute of Technology')


    path = './dataset/CollegeScorecard_Raw_Data'

    col_dtypes = clean_data(path, 'MERGED2009_10_PP.csv')

    df_dict = {uni: None for uni in universities}

    for uni in universities:
        df_dict[uni] = tuition_data(1996, 2017, col_dtypes, uni)
        save_to_dir(uni, 'tuition', df_dict[uni], './dataset/')
