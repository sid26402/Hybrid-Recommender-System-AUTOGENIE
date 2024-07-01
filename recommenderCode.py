import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



# Load the CSV file into a DataFrame
df = pd.read_csv('cars_ds_final_updated.csv')

# Convert 'Ex-Showroom_Price' to numeric type, handling non-numeric values
df['Ex-Showroom_Price'] = pd.to_numeric(df['Ex-Showroom_Price'], errors='coerce')

# Convert 'Length' to numeric type, handling non-numeric values
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')

# Convert 'City_Mileage' to numeric type, handling non-numeric values
df['City_Mileage'] = pd.to_numeric(df['City_Mileage'], errors='coerce')

# Convert 'Displacement' to strings and remove non-numeric characters
df['Displacement'] = df['Displacement'].astype(str).str.replace('[^\d.]', '', regex=True)

# Convert 'Displacement' to numeric type, handling non-numeric values

df['Displacement'] = pd.to_numeric(df['Displacement'], errors='coerce')

# Remove rows with missing values in relevant columns
df.dropna(subset=['Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement'], inplace=True)

# Combine relevant features into a single text-based feature for TF-IDF vectorization
df['Features'] = df['Ex-Showroom_Price'].astype(str) + ' ' + df['Body_Type'].astype(str) + ' ' + df['Length'].astype(str) + ' ' + df['Fuel_Type'].astype(str) + ' ' + df['Displacement'].astype(str)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the 'Features' column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(min_price, max_price, body_type, fuel_type, displacement_min, displacement_max, length_filter, num_recommendations=5):
    # Convert the user's input for 'displacement_min' and 'displacement_max' to numeric
    displacement_min = float(displacement_min)
    displacement_max = float(displacement_max)
    
    # Filter cars based on user preferences for Ex-Showroom Price, Body_Type, Fuel_Type, Displacement, and Length
    if length_filter == 'less':
        filtered_cars = df[(df['Ex-Showroom_Price'] >= min_price) & (df['Ex-Showroom_Price'] <= max_price) &
                           (df['Body_Type'].str.lower().str.strip() == body_type.lower().strip()) &
                           (df['Fuel_Type'].str.lower().str.strip() == fuel_type.lower().strip()) &
                           (df['Displacement'] >= displacement_min) &
                           (df['Displacement'] <= displacement_max) &
                           (df['Length'] < 4000)]
    elif length_filter == 'more':
        filtered_cars = df[(df['Ex-Showroom_Price'] >= min_price) & (df['Ex-Showroom_Price'] <= max_price) &
                           (df['Body_Type'].str.lower().str.strip() == body_type.lower().strip()) &
                           (df['Fuel_Type'].str.lower().str.strip() == fuel_type.lower().strip()) &
                           (df['Displacement'] >= displacement_min) &
                           (df['Displacement'] <= displacement_max) &
                           (df['Length'] >= 4000)]
    else:
        # If 'length_filter' is not 'less' or 'more', return an empty DataFrame
        return pd.DataFrame(columns=['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']), 0

    num_available_cars = len(filtered_cars)

    if num_available_cars == 0:
          # Stop execution if no cars match the criteria
        return pd.DataFrame(columns=['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']), num_available_cars  # Return an empty DataFrame if no cars match the criteria

    # Extract and display the 'Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', and 'Displacement' columns
    recommended_cars = filtered_cars[['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']]

    return recommended_cars, num_available_cars

# User input for each feature
user_min_price = float(input("Enter your minimum preferred Ex-Showroom Price: "))
user_max_price = float(input("Enter your maximum preferred Ex-Showroom Price: "))
user_body_type = input("Enter your preferred Body Type (e.g., SUV, Sedan, Hatchback, MPV): ")
user_fuel_type = input("Enter your preferred Fuel Type (e.g., Diesel, Petrol, Electric): ")
user_displacement_min = float(input("Enter your minimum preferred Displacement: "))
user_displacement_max = float(input("Enter your maximum preferred Displacement: "))
user_length_filter = input("Enter 'less' or 'more' for Length filter: ")

# Get recommendations and the number of available cars within the specified criteria
recommendations, num_available_cars = get_recommendations(user_min_price, user_max_price, user_body_type, user_fuel_type, user_displacement_min, user_displacement_max, user_length_filter)

print(f"Number of Cars Available within the Price Range and Length Filter: {num_available_cars}")
print("Recommended Cars:")
print(recommendations[['Make', 'Model', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])

import pandas as pd

# Load the car_ratings.csv dataset
car_ratings_df = pd.read_csv('maintenance.csv')

sort_column = input("Enter the column for sorting (e.g., 'Avg_Maintenance', 'Avg_KM', 'Safety_Ratings','Nearby_Service_Centers','Reliability'): ")
if sort_column=="Safety_Ratings"or sort_column=="Nearby_Service_Centers" or sort_column=="Reliability" or sort_column=="Avg_KM":
# Extract the recommended cars from the content-based recommendations
    content_recommendations, _ = get_recommendations(user_min_price, user_max_price, user_body_type, user_fuel_type, user_displacement_min, user_displacement_max, user_length_filter)

# Filter the ratings for the recommended models
    recommended_models = content_recommendations['Model']
    ratings_for_recommended_models = car_ratings_df[car_ratings_df['Model'].isin(recommended_models)]

# Merge the content-based recommendations with ratings
    merged_recommendations = content_recommendations.merge(ratings_for_recommended_models, on='Model', how='left')

# Sort the ratings in descending order
    sorted_ratings = merged_recommendations.sort_values(by=sort_column, ascending=False)

# Print the sorted ratings including all columns from content-based recommendations
#print(sorted_ratings[['Make', 'Model','Safety_Ratings', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
# Print the sorted ratings including all columns from content-based recommendations and 'Safety_Ratings'
    print(sorted_ratings[['Model', sort_column, 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
elif sort_column=="Avg_Maintenance":
    content_recommendations, _ = get_recommendations(user_min_price, user_max_price, user_body_type, user_fuel_type, user_displacement_min, user_displacement_max, user_length_filter)

# Filter the ratings for the recommended models
    recommended_models = content_recommendations['Model']
    ratings_for_recommended_models = car_ratings_df[car_ratings_df['Model'].isin(recommended_models)]

# Merge the content-based recommendations with ratings
    merged_recommendations = content_recommendations.merge(ratings_for_recommended_models, on='Model', how='left')

# Sort the ratings in descending order
    sorted_ratings = merged_recommendations.sort_values(by='Avg_Maintenance', ascending=False)

# Print the sorted ratings including all columns from content-based recommendations
#print(sorted_ratings[['Make', 'Model','Safety_Ratings', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
# Print the sorted ratings including all columns from content-based recommendations and 'Safety_Ratings'
    print(sorted_ratings[['Model', 'Avg_Maintenance', 'Ex-Showroom_Price', 'Body_Type', 'Length', 'Fuel_Type', 'Displacement']])
sorted_ratings.to_csv('content_based_recommendations.csv', index=False)
print("Content-based recommendations have been saved to 'content_based_recommendations.csv'.")
import pandas as pd

# Load the content-based recommendations from the CSV file
content_based_recommendations = pd.read_csv('content_based_recommendations.csv')

# Check the number of entries in the DataFrame
num_entries = len(content_based_recommendations)

# Define the percentage to remove based on the number of entries
if num_entries > 20:
    percentage_to_remove = 0.30  # Remove 30% of lower entries
elif num_entries >= 10:
    percentage_to_remove = 0.20  # Remove 20% of lower entries
elif num_entries >= 5:
    percentage_to_remove = 0.10  # Remove 10% of lower entries
else:
    percentage_to_remove = 0  # No removal if fewer than 5 entries

if percentage_to_remove > 0:
    # Sort the DataFrame by the column you want to use as criteria (e.g., 'column_name')

    # Calculate the number of entries to remove
    num_to_remove = int(percentage_to_remove * num_entries)

    # Drop the specified number of lower entries
    content_based_recommendations = content_based_recommendations.iloc[num_to_remove:]

# Save the modified DataFrame back to the CSV file
content_based_recommendations.to_csv('content_based_recommendations.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Load the user-car interaction data from "allinOne.csv"
user_car_data = pd.read_csv("allinOne.csv")

# Load the content-based recommendations from the previous code
content_based_recommendations = pd.read_csv('content_based_recommendations.csv')

# Create a mapping of users to numeric IDs
user_mapping = {user: i for i, user in enumerate(user_car_data['Users'].unique())}

# Create a mapping of car models to numeric IDs
car_mapping = {car: i for i, car in enumerate(content_based_recommendations['Model'].unique())}

# Create a user-item matrix with user ratings
user_item_matrix = np.zeros((len(user_mapping), len(car_mapping)))

# Fill the user-item matrix with ratings from user-car data
for _, row in user_car_data.iterrows():
    user_id = user_mapping[row['Users']]
    car = row['Model']
    if car in car_mapping:  # Check if the car is in the car_mapping dictionary
        car_id = car_mapping[car]
        rating = row['Rating']
        user_item_matrix[user_id, car_id] = rating

# Apply Truncated SVD for matrix factorization with the number of components matching unique car models
svd = TruncatedSVD(n_components=len(car_mapping), random_state=17)

# Factorize the user-item matrix to obtain user factors and car factors
user_factors = svd.fit_transform(user_item_matrix)
car_factors = svd.components_

# Predict user ratings for all car models, including those unique in content_based_recommendations.csv
predicted_ratings = np.dot(user_factors, car_factors)

# Clip the predicted ratings to the 0 to 5 range
predicted_ratings = np.clip(predicted_ratings, 0, 5)

# Create a DataFrame for the predicted ratings with car model columns
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=car_mapping.keys())

# Calculate the average predicted rating for each car model
average_predicted_ratings = predicted_ratings_df.mean()

# Create a DataFrame for the final output with ratings rounded to 2 decimal places
final_output = pd.DataFrame({'Model': average_predicted_ratings.index, 'Predicted_Rating': average_predicted_ratings.values})
final_output['Predicted_Rating'] = final_output['Predicted_Rating'].round(2)

# Sort the final output by 'Predicted_Rating' in descending order
final_output = final_output.sort_values(by='Predicted_Rating', ascending=False)

# Print the final output
print(final_output)

