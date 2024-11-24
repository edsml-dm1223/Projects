import os

from collections.abc import Sequence
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from scipy.stats import randint, uniform

from sklearn import set_config
set_config(transform_output="pandas")

# CLEAN
from .geo import *  # noqa: F401, F403
from .geo import WGS84toOSGB36, get_gps_lat_long_from_easting_northing
from .models import *

__all__ = [
    "Tool",
    "_data_dir",
    "_example_dir",
    "flood_class_from_postcode_methods",
    "flood_class_from_location_methods",
    "house_price_methods",
    "local_authority_methods",
    "historic_flooding_methods",
]

_data_dir = os.path.join(os.path.dirname(__file__), "resources")
_example_dir = os.path.join(os.path.dirname(__file__), "example_data")


# dictionaries with keys of short name and values of long name of
# classification/regression methods


# You should add your own methods here
flood_class_from_postcode_methods = {
    "flood_risk_classifier": "flood_risk_classifier",
}
flood_class_from_location_methods = {
    "flood_risk_classifier": "flood_risk_classifier",
}
historic_flooding_methods = {
    "historic_flooding_classifier": "historic_flooding_classifier",
}
house_price_methods = {
    "house_regressor": "house_regressor",
}
local_authority_methods = {
    "local_authority_classifier": "local_authority_classifier",
}

IMPUTATION_CONSTANTS = {
    "soilType": "Unsurveyed/Urban",
    "elevation": 60.0,
    "nearestWatercourse": "",
    "distanceToWatercourse": 80,
    "localAuthority": np.nan,
}


class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(
        self,
        labelled_unit_data: str = "",
        unlabelled_unit_data: str = "",
        sector_data: str = "",
        district_data: str = "",
        additional_data: dict = {},
    ):
        """
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additional .csv files containing addtional
            information on households.
        """

        # Set defaults if no inputs provided
        if labelled_unit_data == "":
            labelled_unit_data = os.path.join(_data_dir, "postcodes_labelled.csv")

        if unlabelled_unit_data == "":
            unlabelled_unit_data = os.path.join(
                _example_dir, "postcodes_unlabelled.csv"
            )

        if sector_data == "":
            sector_data = os.path.join(_data_dir, "sector_data.csv")
        if district_data == "":
            district_data = os.path.join(_data_dir, "district_data.csv")

        self._postcodedb = pd.read_csv(labelled_unit_data).drop_duplicates()
        self._postcodedb["postcode"] = self._postcodedb["postcode"].apply(
            self.standardise_UK_postcode
        )

        self._postcodedb = self.impute_missing_values(self._postcodedb)

        self._unlabelled_postcodes = pd.read_csv(unlabelled_unit_data).drop_duplicates()
        self._unlabelled_postcodes["postcode"] = self._unlabelled_postcodes["postcode"].apply(
            self.standardise_UK_postcode
        )

        self._sector_data = pd.read_csv(sector_data).drop_duplicates()
        self._sector_data["postcodeSector"] = self._sector_data["postcodeSector"].apply(
            self.standardise_UK_sector
        )

        self.models = Models()

        self.trained_models = {}

        self.le = LabelEncoder()

    def fit(
        self,
        models: List[str] = [],
        update_labels: str = "",
        update_hyperparameters: bool = False,
        **kwargs,
    ):
        """Fit/train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to fit/train
        update_labels : str, optional
            Filename of a .csv file containing an updated
            labelled set of samples
            in the same format as the original labelled set.

            If not provided, the data set provided at
            initialisation is used.
        update_hyperparameters : bool, optional
            If True, models may tune their hyperparameters, where
            possible. If False, models will use their default hyperparameters.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.fit(fcp_methods, update_labels='new_labels.csv')  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        """

        X_train = self._postcodedb.drop(columns=["postcode", "riskLabel", "historicallyFlooded", "medianPrice"])

        if update_labels:
            print("updating labelled sample file")
            self._postcodedb = pd.read_csv(update_labels)

        for model_key in models:
            if model_key in flood_class_from_postcode_methods:
                self.trained_models[model_key] = self.models.flood_risk_model()

                if update_hyperparameters:
                    param_grid = {
                        'classifier__n_estimators': randint(50, 200),
                        'classifier__max_depth': randint(3, 10),
                        'classifier__learning_rate': uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10, 
                        scoring='accuracy',
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42
                    )

                    random_search.fit(X_train, self.le.fit_transform(self._postcodedb["riskLabel"]))
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(X_train, self.le.fit_transform(self._postcodedb["riskLabel"]))

            elif model_key in flood_class_from_location_methods:
                self.trained_models[model_key] = self.models.flood_risk_model()

                if update_hyperparameters:
                    param_grid = {
                        'classifier__n_estimators': randint(50, 200),
                        'classifier__max_depth': randint(3, 10),
                        'classifier__learning_rate': uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10, 
                        scoring='accuracy',
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42
                    )

                    random_search.fit(X_train, self.le.fit_transform(self._postcodedb["riskLabel"]))
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(X_train, self.le.fit_transform(self._postcodedb["riskLabel"]))

            elif model_key in historic_flooding_methods:
                self.trained_models[model_key] = self.models.historic_flooding_model()

                if update_hyperparameters:
                    param_grid = {
                        'classifier__n_estimators': randint(50, 200),
                        'classifier__max_depth': randint(3, 10),
                        'classifier__learning_rate': uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10, 
                        scoring='accuracy',
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42
                    )

                    random_search.fit(X_train, self._postcodedb["historicallyFlooded"])
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(X_train, self._postcodedb["historicallyFlooded"])

            elif model_key in house_price_methods:
                self.trained_models[model_key] = self.models.house_price_model()

                if update_hyperparameters:
                    param_grid = {
                        'regressor__n_estimators': [100, 200, 300],
                        'regressor__max_depth': [3, 5, 10],
                        'regressor__learning_rate': [0.01, 0.1],
                        'regressor__subsample': [0.8, 1.0],
                        'regressor__colsample_bytree': [0.8, 1.0],
                        'regressor__reg_alpha': [0, 0.1],
                        'regressor__reg_lambda': [1, 2]
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10,
                        cv=5,
                        scoring='neg_root_mean_squared_error', 
                        random_state=42,
                        verbose=0,
                        n_jobs=-1
                    )

                    random_search.fit(X_train, self._postcodedb["medianPrice"])
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(X_train, self._postcodedb["medianPrice"])

            elif model_key in local_authority_methods:
                self.trained_models[model_key] = self.models.local_authority_modelX_train()

                if update_hyperparameters:
                    param_grid = {
                        'classifier__n_estimators': randint(50, 200),
                        'classifier__max_depth': randint(3, 10),
                        'classifier__learning_rate': uniform(0.01, 0.2),
                    }

                    random_search = RandomizedSearchCV(
                        estimator=self.trained_models[model_key],
                        param_distributions=param_grid,
                        n_iter=10, 
                        scoring='accuracy',
                        cv=5,
                        verbose=0,
                        n_jobs=-1,
                        random_state=42
                    )

                    random_search.fit(X_train, self._postcodedb["localAuthority"])
                    self.trained_models[model_key] = random_search.best_estimator_
                else:
                    self.trained_models[model_key].fit(X_train, self._postcodedb["localAuthority"])
            else:
                raise ValueError(f"Unknown model key: {model_key}")

    def lookup_easting_northing(
        self, postcodes: Sequence, dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """Get a dataframe of OS eastings and northings from a sequence of
        input postcodes in the labelled or unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the easting and northing columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of 'easthing' and 'northing',
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the available postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['RH16 2QE'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        >>> results = tool.lookup_easting_northing(['RH16 2QE', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        AB1 2PQ        NaN       NaN
        """

        postcodes = pd.Index(postcodes)

        frame = self._postcodedb.copy()
        frame = frame.set_index("postcode")
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ["easting", "northing"]].astype(dtype)

    def lookup_lat_long(
        self, postcodes: Sequence, dtype: np.dtype = np.float64
    ) -> pd.DataFrame:
        """Get a Pandas dataframe containing GPS latitude and longitude
        information for a sequence of postcodes in the labelled or
        unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the latitude and longitude columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NaNs in the latitude
            and longitude columns.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        """
        # Standardise the postcodes
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        easting_northing_df = self.lookup_easting_northing(postcodes)

        if easting_northing_df.empty:
            return pd.DataFrame(
                columns=["longitude", "latitude"], index=postcodes, dtype=dtype
            )

        # Convert the easting and northing to latitude and longitude
        lat, long = get_gps_lat_long_from_easting_northing(
            easting_northing_df["easting"],
            easting_northing_df["northing"],
            rads=False,
            dms=False,
            dtype=dtype,
        )

        # Return the lat and long in a dataframe
        return pd.DataFrame({"latitude": lat, "longitude": long}, index=postcodes)

    def impute_missing_values(
        self, data: pd.DataFrame,
        method: str = 'knn',
        n_neighbors: int = 4,
        constant_values: dict = IMPUTATION_CONSTANTS,
    ) -> pd.DataFrame:
        """Impute missing values in a dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            DataFrame (in the format of the unlabelled postcode data)
            potentially containing missing values as NaNs, or with missing
            columns.

        method : str, optional
            Method to use for imputation. Options include:
            - 'mean', to use the mean for the labelled dataset
            - 'constant', to use a constant value for imputation
            - 'knn' to use k-nearest neighbours imputation from the
              labelled dataset

        constant_values : dict, optional
            Dictionary containing constant values to
            use for imputation in the format {column_name: value}.
            Only used if method is 'constant'.

        Returns
        -------

        pandas.DataFrame
            DataFrame with missing values imputed.

        Examples
        --------

        >>> tool = Tool()
        >>> missing = os.path.join(_example_dir, 'postcodes_missing_data.csv')
        >>> data = pd.read_csv(missing)
        >>> data = tool.impute_missing_values(data)  # doctest: +SKIP
        """
        if method not in ['mean', 'constant', 'knn']:
            raise ValueError(f"Unsupported method '{method}'. Choose from 'mean', 'constant', or 'knn'.")
        
        dataframe = data.copy()
        if method == 'mean':
            numeric_col = dataframe.select_dtypes(include=[np.number]).columns
            dataframe[numeric_col] = dataframe[numeric_col].fillna(dataframe[numeric_col].mean(numeric_only=True))
        elif method == 'constant'                                                                                                                                                                                                       
            if constant_values is None:
                raise ValueError("Constant values must be provided for 'constant' imputation.")
            for col, value in constant_values.items():
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].fillna(value)
        elif method == 'knn':
            cat_col = dataframe.select_dtypes(include=[object]).columns
            le = OrdinalEncoder()
            dataframe[cat_col] = le.fit_transform(dataframe[cat_col])
            imputer = KNNImputer(n_neighbors=n_neighbors)
            dataframe_imputed = imputer.fit_transform(dataframe)
            
            # Decode back to original categories
            imputed_decoded = le.inverse_transform(dataframe_imputed[cat_col])
            dataframe_imputed[cat_col] = imputed_decoded
        return dataframe_imputed
    


    def predict_flood_class_from_postcode(
        self, postcodes: Sequence[str], method: str = "flood_risk_classifier"
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
            Returns NaN for postcode units not in the available postcode files.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        if method == "all_zero_risk":
            return pd.Series(
                data=np.ones(len(postcodes), int),
                index=np.asarray(postcodes),
                name="riskLabel",
            )
        
        elif method == "flood_risk_classifier":
            # Get the rows in the unlabelled data corresponding to `postcodes`
            df_postcodes = self._unlabelled_postcodes[
                self._unlabelled_postcodes["postcode"].isin(postcodes)
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            y_pred = self.le.inverse_transform(y_pred)

            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="riskLabel",
                dtype="int64"
            )

        else:
            raise NotImplementedError(f"method {method} not implemented")

    def predict_flood_class_from_OSGB36_location(
        self,
        eastings: Sequence[float],
        northings: Sequence[float],
        method: str = "flood_risk_classifier",
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        """

        if method == "all_zero_risk":
            return pd.Series(
                data=np.ones(len(eastings), int),
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
            )
        elif method == "flood_risk_classifier":

            df_postcodes = self._unlabelled_postcodes[
                (self._unlabelled_postcodes["easting"].isin(eastings))
                & (self._unlabelled_postcodes["northing"].isin(northings))
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            y_pred = self.le.inverse_transform(y_pred)

            return pd.Series(
                data=y_pred,
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name="riskLabel",
                dtype="int64"
            )
        else:
            raise NotImplementedError(f"method {method} not implemented")

    def predict_flood_class_from_WGS84_locations(
        self,
        longitudes: Sequence[float],
        latitudes: Sequence[float],
        method: str = "flood_risk_classifier",
    ) -> pd.Series:
        """
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : str (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels multi-indexed by
            location as a (longitude, latitude) pair.
        """

        idx = pd.MultiIndex.from_tuples(
            [(lng, lat) for lng, lat in zip(longitudes, latitudes)]
        )

        if method == "all_zero_risk":
            return pd.Series(
                data=np.ones(len(longitudes), int),
                index=idx,
                name="riskLabel",
            )
        elif method == "flood_risk_classifier":

            eastings, northings = WGS84toOSGB36(latitudes, longitudes)

            df_postcodes = self._unlabelled_postcodes[
                (self._unlabelled_postcodes["easting"].isin(eastings))
                & (self._unlabelled_postcodes["northing"].isin(northings))
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            y_pred = self.le.inverse_transform(y_pred)

            return pd.Series(
                data=y_pred,
                index=idx,
                name="riskLabel",
                dtype="int64"
            )
        else:
            raise NotImplementedError(f"method {method} not implemented")

    def predict_median_house_price(
        self, postcodes: Sequence[str], method: str = "house_regressor"
    ) -> pd.Series:
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        if method == "all_england_median":
            return pd.Series(
                data=np.full(len(postcodes), 245000.0),
                index=np.asarray(postcodes),
                name="medianPrice",
            )

        elif method == "house_regressor":
            # Get the rows in the unlabelled data corresponding to `postcodes`
            df_postcodes = self._unlabelled_postcodes[
                self._unlabelled_postcodes["postcode"].isin(postcodes)
            ]
        

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)
            
            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="medianPrice",
            )

        else:
            raise NotImplementedError(f"method {method} not implemented")

    def predict_local_authority(
        self,
        eastings: Sequence[float],
        northings: Sequence[float],
        method: str = "local_authority_classifier",
    ) -> pd.Series:
        """
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            locations, and multiindexed by the location as a
            (easting, northing) tuple.
        """
        idx = pd.MultiIndex.from_tuples(
            [(est, nth) for est, nth in zip(eastings, northings)]
        )

        if method == "all_nan":

            return pd.Series(
                data=np.full(len(eastings), np.nan),
                index=idx,
                name="localAuthority",
            )

        elif method == "local_authority_classifier":
            # Get the rows in the unlabelled data corresponding to `postcodes`
            df_postcodes = self._unlabelled_postcodes[
                (self._unlabelled_postcodes["easting"].isin(eastings))
                & (self._unlabelled_postcodes["northing"].isin(northings))
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            return pd.Series(
                data=y_pred,
                index=idx,
                name="localAuthority",
            )

        else:
            raise NotImplementedError(f"method {method} not implemented")

    def predict_historic_flooding(
        self, postcodes: Sequence[str], method: str = "historic_flooding_classifier"
    ) -> pd.Series:
        """
        Generate series predicting whether a collection of postcodes
        has experienced historic flooding.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        if method == "all_false":
            return pd.Series(
                data=np.full(len(postcodes), False),
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )

        elif method == "historic_flooding_classifier":
            # Get the rows in the unlabelled data corresponding to `postcodes`
            df_postcodes = self._unlabelled_postcodes[
                self._unlabelled_postcodes["postcode"].isin(postcodes)
            ]

            # Impute missing values
            df_postcodes = self.impute_missing_values(df_postcodes)

            self.fit([method], update_hyperparameters=True)

            y_pred = self.trained_models[method].predict(
                df_postcodes.drop(columns=["postcode"])
            )

            return pd.Series(
                data=y_pred,
                index=np.asarray(postcodes),
                name="historicallyFlooded",
            )

        else:
            raise NotImplementedError(f"method {method} not implemented")

    def estimate_total_value(self, postal_data: Sequence[str]) -> pd.Series:
        """
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        The estimate is based on the median house price for the area and an
        estimate of the number of properties it contains.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcode sectors (either
            may be used).


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """
        # TODO: make sure this can take either psotcode or sector

        postal_data = [self.standardise_UK_postcode(pc) for pc in postal_data]

        # unlabelled_postcodes_with_households = self._unlabelled_postcodes
        
        # unlabelled_postcodes_with_households["sector"] = unlabelled_postcodes_with_households["postcode"].apply(
        #     lambda x: x[:-2]
        # )
    
        # # merge the households column of self._sector_data into self._unlabelled_postcodes for the corresponding sector in the sector column
        # unlabelled_postcodes_with_households = unlabelled_postcodes_with_households.merge(
        #     self._sector_data[['postcodeSector', 'households', 'numberOfPostcodeUnits']],
        #     left_on='sector',
        #     right_on='postcodeSector',
        #     how='left'
        # )

        # predict median house price for given postcodes
        median_house_prices = self.predict_median_house_price(postal_data)

        postcode_sectors = [postcode[:-2] for postcode in postal_data]

        filtered_data = self._sector_data[self._sector_data['postcodeSector'].isin(postcode_sectors)]

        matching_households = np.array(filtered_data['households'])
        matching_postcode_units = np.array(filtered_data['numberOfPostcodeUnits']) + 1

        # print(median_house_prices.values)
        # print(type(median_house_prices.values))
        # print(matching_households)
        # print(type(matching_households))
        # print(matching_postcode_units)
        # print(type(matching_postcode_units))

        return pd.Series(
            data=median_house_prices.values * matching_households / matching_postcode_units,
            index=np.asarray(postal_data),
            name="totalValue",
        )


    def estimate_annual_human_flood_risk(
        self, postcodes: Sequence[str], risk_labels: Union[pd.Series, None] = None
    ) -> pd.Series:
        """
        Return a series of estimates of the risk to human life for a
        collection of postcodes.

        Risk is defined here as an impact coefficient multiplied by the
        estimated number of people under threat multiplied by the probability
        of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual human flood risk estimates
            indexed by postcode.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(postcodes).values

        if len(risk_labels) != len(postcodes):
            raise ValueError("risk_labels must be the same length as postcodes")

        risk_to_probability = {
            7: 0.05,
            6: 0.03,
            5: 0.02,
            4: 0.01,
            3: 0.005,
            2: 0.002,
            1: 0.001,
        }

        postcode_sectors = [postcode[:-2] for postcode in postcodes]

        matching_data = self._sector_data.loc[
            self._sector_data['postcodeSector'].isin(postcode_sectors),
            ['headcount', 'numberOfPostcodeUnits']
        ]
      
        matching_headcount = np.array(matching_data['headcount'])
        matching_postcode_units = np.array(matching_data['numberOfPostcodeUnits']) + 1

        return pd.Series(
            data=[
            0.1 * (headcount / units) * risk_to_probability[risk_label]
            for headcount, units, risk_label in zip(matching_headcount, matching_postcode_units, risk_labels)
            ],
            index=np.asarray(postcodes),
            name="humanRisk",
        )

    def estimate_annual_flood_economic_risk(
        self, postcodes: Sequence[str], risk_labels: Union[pd.Series, None] = None
    ) -> pd.Series:
        """
        Return a series of estimates of the total economic property risk
        for a collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            optionally provide a Pandas Series containing flood risk
            classifiers, as predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual economic flood risk estimates indexed
            by postcode.
        """
        postcodes = [self.standardise_UK_postcode(pc) for pc in postcodes]

        risk_labels = risk_labels or self.predict_flood_class_from_postcode(postcodes).values

        if len(risk_labels) != len(postcodes):
            raise ValueError("risk_labels must be the same length as postcodes")

        risk_to_probability = {
            7: 0.05,
            6: 0.03,
            5: 0.02,
            4: 0.01,
            3: 0.005,
            2: 0.002,
            1: 0.001,
        }

        total_values = self.estimate_total_value(postcodes)

        return pd.Series(
            data=[0.5 * value * risk_to_probability[risk_label] for value, risk_label in zip(total_values.values, risk_labels)],
            index=np.asarray(postcodes),
            name="economicRisk",
        )

    
    def standardise_UK_postcode(self, postcode: str, is_sector=False) -> str:
        """
        Standardise a postcode to upper case and ensure it has a space in the middle.

        This will also work for sectors

        Parameters
        ----------

        postcode : str
            Postcode to standardise.

        Returns
        -------

        str
            Standardised postcode.
        """
        postcode = postcode.replace(" ", "").upper()
        if len(postcode) > 3:
            return postcode[:-3] + " " + postcode[-3:]
        return postcode
    
    def standardise_UK_sector(self, postcode: str) -> str:
        """
        Standardise a postcode to upper case and ensure it has a space in the middle.

        This will also work for sectors

        Parameters
        ----------

        postcode : str
            Postcode to standardise.

        Returns
        -------

        str
            Standardised postcode.
        """
        postcode = postcode.replace(" ", "").upper()
        if len(postcode) > 3:
            return postcode[:-1] + " " + postcode[-1:]
        return postcode
    
    def make_output(self, postcodes=None, eastings=None, northings=None, longitudes=None, latitudes=None, path = "output/output.csv"):
        """
        Concatenate the outputs of all predict_* methods.

        Parameters
        ----------
        postcodes : sequence of strs, optional
            Sequence of postcode units.
        eastings : sequence of floats, optional
            Sequence of OSGB36 eastings.
        northings : sequence of floats, optional
            Sequence of OSGB36 northings.
        longitudes : sequence of floats, optional
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats, optional
            Sequence of WGS84 latitudes.

        Returns
        -------
        pandas.DataFrame
            Combined output from all prediction functions.
        """
        output = pd.DataFrame() 

        if postcodes is not None:
            flood_risk_postcode = self.predict_flood_class_from_postcode(postcodes).to_frame()
            house_prices = self.predict_median_house_price(postcodes).to_frame()
            historical_flood = self.predict_historic_flooding(postcodes).to_frame()
            output = pd.concat([flood_risk_postcode, house_prices, historical_flood], axis=1)

        if eastings is not None and northings is not None:
            flood_risk_osgb36 = self.predict_flood_class_from_OSGB36_location(eastings, northings).to_frame()
            local_authority = self.predict_local_authority(eastings, northings).to_frame()
            output = pd.concat([flood_risk_osgb36, local_authority], axis=1)

        if longitudes is not None and latitudes is not None:
            flood_risk_wgs84 = self.predict_flood_class_from_WGS84_locations(longitudes, latitudes)
            output = flood_risk_wgs84.to_frame()

        output.to_csv(path)


