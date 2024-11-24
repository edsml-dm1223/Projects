from typing import List

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, FunctionTransformer, LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor, XGBClassifier

from sklearn import set_config
set_config(transform_output="pandas")


from .geo import OSGB36toWGS84

class Models:
    """
    A class to build machine learning pipelines for postcode-related predictions.
 
    This class provides methods to create machine learning models for tasks such as
    flood risk prediction, historic flooding classification, house price estimation,
    and local authority prediction. Each method includes preprocessing steps tailored
    to the specific task.
 
    Attributes
    ----------
    cat_pipeline : sklearn.pipeline.Pipeline
        A preprocessing pipeline for categorical data using `OrdinalEncoder`.
    """

    def __init__(self) -> None:
        """
        Initialize the Models class with a categorical preprocessing pipeline.
        """
        pass
    
    def flood_risk_model(self) -> str:
        """
        Create a pipeline to predict flood risk using a random forest classifier.
 
        This method preprocesses the input data by encoding categorical features and
        optionally dropping specified columns, then applies a Random Forest Classifier
        for predictions.
 
        Parameters
        ----------
        data : pd.DataFrame
            The input data to train the model, containing both features and target.
        cols : list of str, optional
            Columns to drop from the input data (default is an empty list).
 
        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object that preprocesses the input data and predicts flood risk.
 
        Examples
        --------
        >>> models = Models()
        >>> data = pd.DataFrame({'feature1': ['A', 'B'], 'feature2': [1, 2]})
        >>> pipeline = models.flood_risk_model(data, cols=['feature2'])
        >>> type(pipeline)
        <class 'sklearn.pipeline.Pipeline'>
        """

        preprocessor = Pipeline([
            ('drop_cols', FunctionTransformer(
                lambda X: self._drop_columns(X, ['localAuthority', 'nearestWatercourse', 'soilType']), 
                validate=False)),
            ('scale_numerical', FunctionTransformer(
                lambda X: self._scale_numerical(X, scaling_type="standard"), 
                validate=False)),
            ('cat_encoding', FunctionTransformer(
                self._encode_categoricals, 
                validate=False)),
        ])

        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                ("classifier", XGBClassifier())
            ]
        )
        return pipeline
    

    def historic_flooding_model(self) -> Pipeline:
        """
        Create a pipeline to predict historic flooding using a random forest classifier.
 
        Parameters
        ----------
        data : pd.DataFrame
            The input data to train the model, containing both features and target.
        cols : list of str, optional
            Columns to drop from the input data (default is an empty list).
 
        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object that preprocesses the input data and predicts historic flooding.
 
        Examples
        --------
        >>> models = Models()
        >>> data = pd.DataFrame({'feature1': ['A', 'B'], 'feature2': [1, 2]})
        >>> pipeline = models.historic_flooding_model(data)
        >>> type(pipeline)
        <class 'sklearn.pipeline.Pipeline'>
        """
        preprocessor = Pipeline([
            ('drop_cols', FunctionTransformer(
                lambda X: self._drop_columns(X, ['localAuthority', 'nearestWatercourse', 'soilType']), 
                validate=False)),
            ('scale_numerical', FunctionTransformer(
                lambda X: self._scale_numerical(X, scaling_type="standard"), 
                validate=False)),
            ('cat_encoding', FunctionTransformer(
                self._encode_categoricals, 
                validate=False)),
        ])

        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                ("classifier", XGBClassifier())
            ]
        )
        return pipeline

    def house_price_model(self) -> Pipeline:
        """
        Create a pipeline to estimate house prices using a linear regression model.
 
        Parameters
        ----------
        data : pd.DataFrame
            The input data to train the model, containing both features and target.
        cols : list of str, optional
            Columns to drop from the input data (default is an empty list).
 
        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object that preprocesses the input data and estimates house prices.
 
        Examples
        --------
        >>> models = Models()
        >>> data = pd.DataFrame({'feature1': ['A', 'B'], 'feature2': [1, 2]})
        >>> pipeline = models.house_price_model(data, cols=['feature2'])
        >>> type(pipeline)
        <class 'sklearn.pipeline.Pipeline'>
        """
        preprocessor = Pipeline([
            ('drop_cols', FunctionTransformer(
                lambda X: self._drop_columns(X, ['localAuthority', 'nearestWatercourse', 'soilType']), 
                validate=False)),
            ('scale_numerical', FunctionTransformer(
                lambda X: self._scale_numerical(X, scaling_type="standard"), 
                validate=False)),
            ('cat_encoding', FunctionTransformer(
                self._encode_categoricals, 
                validate=False)),
        ])

        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor())
            ]
        )
        return pipeline

    def local_authority_model(self) -> Pipeline:
        """
        Create a pipeline to predict local authority outcomes using logistic regression.
 
        Parameters
        ----------
        data : pd.DataFrame
            The input data to train the model, containing both features and target.
        cols : list of str, optional
            Columns to drop from the input data (default is an empty list).
 
        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object that preprocesses the input data and predicts local authority outcomes.
 
        Examples
        --------
        >>> models = Models()
        >>> data = pd.DataFrame({'feature1': ['A', 'B'], 'feature2': [1, 2]})
        >>> pipeline = models.local_authority_model(data)
        >>> type(pipeline)
        <class 'sklearn.pipeline.Pipeline'>
        """
        preprocessor = Pipeline([
            ('drop_cols', FunctionTransformer(
                lambda X: self._drop_columns(X, ['localAuthority', 'nearestWatercourse', 'soilType']), 
                validate=False)),
            ('scale_numerical', FunctionTransformer(
                lambda X: self._scale_numerical(X, scaling_type="standard"), 
                validate=False)),
            ('cat_encoding', FunctionTransformer(
                self._encode_categoricals, 
                validate=False)),
        ])

        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                ("classifier", XGBClassifier())
            ]
        )
        return pipeline
    
    def _drop_columns(self, data: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
        """
        Drop specified columns from the given DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame from which columns will be dropped.
        cols_to_drop : list of str
            List of column names to be dropped from the DataFrame.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified columns removed.
        
        Examples
        --------
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        >>> cols_to_drop = ['B', 'C']
        >>> result = self._drop_columns(data, cols_to_drop)
        >>> result
           A
        0  1
        1  2
        2  3
        """
        return data.drop(columns=cols_to_drop)
    
    def _add_lat_long(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the latitude and longitude from the postcode.
        
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing 'easting' and 'northing' columns.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with added 'latitude' and 'longitude' columns.
        
        Examples
        --------
        >>> data = pd.DataFrame({'easting': [651409, 651409], 'northing': [313177, 313177]})
        >>> result = self._add_lat_long(data)
        >>> result
             easting  northing   latitude  longitude
        0    651409    313177  52.657570  -1.717921
        1    651409    313177  52.657570  -1.717921
        """
        latitudes, longitudes = OSGB36toWGS84(data.easting, data.northing)
        data['latitude'] = latitudes
        data['longitude'] = longitudes
        return data

    # def _add_proximity_risk(self, data: pd.DataFrame) -> pd.DataFrame:
    #     proximity_risk = data.distanceToWatercourse / (data.elevation + 1)
    #     data['proximity_risk'] = proximity_risk
    #     return data

    def _add_interaction_terms(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction terms to the given DataFrame by encoding categorical variables and creating combinations.
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the data to which interaction terms will be added.
        Returns
        -------
        pd.DataFrame
            A DataFrame with the added interaction terms.
        Examples
        --------
        >>> data = pd.DataFrame({
        ...     'soilType': ['clay', 'silt', 'sand', 'clay'],
        ...     'elevation': [100, 200, 150, 300]
        ... })
        >>> result = self._add_interaction_terms(data)
        >>> result
           soilType  elevation soilType/Elevation
        0      clay        100          0/Low
        1      silt        200          2/Mid
        2      sand        150          1/Mid
        3      clay        300          0/Very High
        """

        le_soil = LabelEncoder()
        # le_watercourse = LabelEncoder()

        data['soilType_encoded'] = le_soil.fit_transform(data.soilType)
        # data['nearestWatercourse_encoded'] = le_watercourse.fit_transform(data.nearestWatercourse)

        bins_elevation = pd.qcut(data['elevation'], q=4, labels=["Low", "Mid", "High", "Very High"])
        # bins_distance = pd.qcut(data['distanceToWatercourse'], q=4, labels=["Low", "Mid", "High", "Very High"])

        data['soilType/Elevation'] = data['soilType_encoded'].astype(str) + '/' + bins_elevation.astype(str)
        # data['distanceToWatercourse/nearestWatercourse'] = bins_distance.astype(str) + '/' + data['nearestWatercourse_encoded'].astype(str)

        # data.drop(columns=['soilType_encoded', 'nearestWatercourse_encoded'], inplace=True)
        data.drop(columns=['soilType_encoded'], inplace=True)


        return data
    
    def _scale_numerical(self, data: pd.DataFrame, scaling_type: str = "standard") -> pd.DataFrame:
        """
        Scale numerical columns in the given DataFrame using the specified scaling method.
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing numerical columns to be scaled.
        scaling_type : str, optional
            The type of scaling to apply. Options are 'standard', 'minmax', or 'robust'.
            Default is 'standard'.
        Returns
        -------
        pd.DataFrame
            A DataFrame with the numerical columns scaled according to the specified method.
        Raises
        ------
        ValueError
            If an unsupported scaling_type is provided.
        Examples
        --------
        >>> data = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        >>> result = self._scale_numerical(data, scaling_type='minmax')
        >>> result
             A    B
        0  0.0  0.0
        1  0.5  0.5
        2  1.0  1.0
        """

        # Select numerical columns
        num_columns = data.select_dtypes(include=['float64', 'int64']).columns

        # Select the scaler based on scaling_type
        if scaling_type == "standard":
            scaler = StandardScaler()
        elif scaling_type == "minmax":
            scaler = MinMaxScaler()
        elif scaling_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaling_type. Use 'standard', 'minmax', or 'robust'.")

        # Apply scaling to numerical columns
        if not num_columns.empty:
            scaled_values = scaler.fit_transform(data[num_columns])
            scaled_df = pd.DataFrame(scaled_values, index=data.index, columns=num_columns)
            # Replace scaled columns in the original DataFrame
            data[num_columns] = scaled_df

        return data
    
    def _encode_categoricals(self, data):
        """
        Encode categorical columns in the given DataFrame using one-hot encoding.
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing categorical columns to be encoded.
        Returns
        -------
        pd.DataFrame
            A DataFrame with the categorical columns encoded using one-hot encoding.
            Original categorical columns are dropped and replaced with their encoded counterparts.
        Examples
        --------
        >>> data = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': [1, 2, 3]})
        >>> result = self._encode_categoricals(data)
        >>> result
           B  A_b
        0  1    0
        1  2    1
        2  3    0
        """
        
        cat_columns = data.select_dtypes(include=["object", "category"]).columns
        # print(f"Categorical columns to encode: {cat_columns}")
        
        if cat_columns.empty:
            # print("No categorical columns to encode.")
            return data

        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded = encoder.fit_transform(data[cat_columns])
        # print(f"Encoded shape: {encoded.shape}")

        encoded_df = pd.DataFrame(
            encoded, 
            index=data.index, 
            columns=encoder.get_feature_names_out(cat_columns)
        )
        # print(f"Encoded DataFrame columns: {encoded_df.columns}")

        data = data.drop(columns=cat_columns)
        data = pd.concat([data, encoded_df], axis=1)
        return data
