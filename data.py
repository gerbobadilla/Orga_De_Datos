import numpy as np
import pandas as pd
import csv
from time import time

# plots
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import matplotlib.pyplot as plt



def reducir_train():
	data_17_08 = pd.read_csv('../data/properati/properati-AR-2017-08-01-properties-sell.csv')
	data_17_02 = pd.read_csv('../data/properati/properati-AR-2017-02-01-properties-sell.csv')
	data_16_08 = pd.read_csv('../data/properati/properati-AR-2016-08-01-properties-sell.csv')
	data_16_02 = pd.read_csv('../data/properati/properati-AR-2016-02-01-properties-sell.csv')
	data_15_08 = pd.read_csv('../data/properati/properati-AR-2015-08-01-properties-sell.csv')
	data_15_02 = pd.read_csv('../data/properati/properati-AR-2015-02-01-properties-sell.csv')
	data_14_08 = pd.read_csv('../data/properati/properati-AR-2014-08-01-properties-sell.csv')
	data_14_02 = pd.read_csv('../data/properati/properati-AR-2014-02-01-properties-sell.csv')
	data_13_08 = pd.read_csv('../data/properati/properati-AR-2013-08-01-properties-sell.csv')

	data_frames = [data_17_08, data_17_02, data_16_08, data_16_02,data_15_08,data_15_02,data_14_08,data_14_02,data_13_08]
	train = pd.concat(data_frames)

	#Nos quedamos con las zonas de Capital Federal y GBA
	train = train.loc[(train.state_name=='Capital Federal') | (train.state_name.str.contains('G.B.A'))]

	train['created_on'] = pd.to_datetime(train['created_on'])
	train = train.drop('image_thumbnail', axis=1)
	train = train.drop('operation', axis=1)
	train = train.drop('properati_url', axis=1)
	train = train.drop('description', axis=1)
	train = train.drop('place_with_parent_names', axis=1)
	train = train.drop('title', axis=1)
	train = train.drop('place_name', axis=1)
	train = train.drop('state_name', axis=1)
	train = train.drop('property_type', axis=1)
	train = train.drop('lat-lon', axis=1)
	train = train.drop('id', axis=1)
	train = train.drop('country_name', axis=1)
	train = train.drop('extra', axis=1)
	train = train.drop('price_aprox_local_currency', axis=1)
	train = train.drop('price_per_m2', axis=1)
	train = train.drop('currency', axis=1)
	train = train.drop('created_on', axis=1)
	train = train.drop('geonames_id', axis=1)
	train = train.drop('lat', axis=1)
	train = train.drop('lon', axis=1)
	train = train.drop('surface_in_m2', axis=1)
	train = train.drop('price', axis=1)

	train['expenses'] = pd.to_numeric(train['expenses'], errors='coerce')
	
	train[['price_usd_per_m2', 'rooms','surface_covered_in_m2','surface_total_in_m2']] \
	= train[['price_usd_per_m2', 'rooms','surface_covered_in_m2','surface_total_in_m2']].fillna(value=0)
	train[['expenses']] = train[['expenses']].fillna(value=0)
	train[['floor']] = train[['floor']].fillna(value=0)
	train[['price_aprox_usd']] = train[['price_aprox_usd']].fillna(value=0)
	train=train.ix[train['price_aprox_usd'] > 0]

	return train


def reducir_test():
	# cargamos el archivo de test
	test  = pd.read_csv("../data/properati/properati_dataset_testing_noprice.csv")

	test = test.drop('description', axis=1)
	test = test.drop('place_with_parent_names', axis=1)
	test = test.drop('place_name', axis=1)
	test = test.drop('property_type', axis=1)
	test = test.drop('lat-lon', axis=1)
	test = test.drop('country_name', axis=1)
	test = test.drop('operation', axis=1)
	test = test.drop('created_on', axis=1)
	test = test.drop('state_name', axis=1)
	test = test.drop('lat', axis=1)
	test = test.drop('lon', axis=1)

	test['expenses'] = pd.to_numeric(test['expenses'], errors='coerce')

	test[['surface_total_in_m2','surface_covered_in_m2','floor','rooms','expenses']] = \
	test[['surface_total_in_m2','surface_covered_in_m2','floor','rooms','expenses']].fillna(value=0)
	test[['expenses']] = test[['expenses']].fillna(value=0)

	return test


# funcion para generar el archivo que se sube a kaggle

def write_submission(test_data, prediction, file_output):
    
    archivo_entrada = open(test_data)
    entrada_csv = csv.reader(archivo_entrada)
    next(entrada_csv, None)  # skip the headers

    archivo_salida = open(file_output, 'w')
    submit_csv = csv.writer(archivo_salida)
    submit_csv.writerow(['id', 'price_usd'])

    for reg1, reg2  in zip(entrada_csv, prediction):
        linea = [reg1[0], round(reg2, 2)]
        submit_csv.writerow(linea)
    archivo_salida.close()
