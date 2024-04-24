from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

class PredictView(APIView):
    def post(self, request):
        # Extract input values from request
        v1 = request.data.get('v1')
        v2 = request.data.get('v2')
        v3 = request.data.get('v3')
        v4 = request.data.get('v4')
        v5 = request.data.get('v5')
        
        # Check if any value is None
        if None in [v1, v2, v3, v4, v5]:
            return Response({'error': 'One or more values are missing'}, status=status.HTTP_400_BAD_REQUEST)

        # Convert values to float
        try:
            v1 = float(v1)
            v2 = float(v2)
            v3 = float(v3)
            v4 = float(v4)
            v5 = float(v5)
        except ValueError:
            return Response({'error': 'One or more values are not valid numbers'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the trained model and preprocess data
        data = pd.read_csv('F:\\MLOPS\\backend\\Student_Performance.csv')  # Update with your file path
        data = data.dropna()
        l = LabelEncoder()
        data["Extracurricular_Activities"] = l.fit_transform(data["Extracurricular_Activities"])
        x = data.iloc[:, [0, 1, 2, 3, 4]]
        y = data.iloc[:, -1]
        
        # Train the model
        rg = LinearRegression()
        rg.fit(x, y)

        # Make predictions
        out = rg.predict(np.array([v1, v2, v3, v4, v5]).reshape(1, -1))
        prediction = int(out[0])

        # Return the prediction
        return Response({'prediction': prediction}, status=status.HTTP_200_OK)
