import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import matplotlib.ticker as mtick
from scipy import stats
from sklearn.cluster import KMeans


data = pd.read_csv('churn.csv')
bf =pd.DataFrame(data)

"""============================data details============================"""

print("====================df head=======================================")
print(bf.head())
print("====================df shape=======================================")
print(bf.shape)
print("=====================columns======================================")
print(bf.columns)
print("======================dtypes=====================================")
print(bf.dtypes)
print("=======================info====================================")
print(bf.info)
print("=======================description====================================")
print(bf.describe())
print("========================null values===================================")
print(bf.isnull().sum())
print("======================unique values=====================================")
print(bf.nunique())
print("====================df columns===============================")
print(bf.columns)

"""==============================DATA PREPROCESS====================================="""

#we don't have to convert 'customer_id' because it is hash values

df = bf.drop(['customer_id'], axis=1)

"""converting & encoding data"""

for col in df.columns:
    if df[col].dtype == 'object':  # Categorical data
        df[col] = LabelEncoder().fit_transform(df[col])
    elif df[col].dtype== 'bool':  # Boolean data
        df[col] = df[col].astype(int)

"""adding customer_id because we use later"""

df['customer_id'] = bf['customer_id']

"""=============================MODEL PREDICTION, FEATURE IMPORTANCE & METRIX=========================="""

#we dropping customer id because it is hash value

x = df.drop(['churn','customer_id'], axis=1)
y = df['churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

"""selecting feature"""

target_importance  = model.feature_importances_

imp_df = pd.DataFrame({'features':x.columns,
                        'rf_imp':target_importance})
sorted_imp_df = imp_df.sort_values(by='rf_imp', ascending=False)
print(sorted_imp_df)

threshold_value = 0.0 #there is no overfitting in this data so I reduce the threshold

selected_feature = imp_df[(imp_df['rf_imp'])>=threshold_value]['features'].tolist()
print("-------------------")
print(selected_feature)

# Create a mask to filter the selected features from the training and test sets
selected_indices = [x.columns.tolist().index(f) for f in selected_feature]
x_train_selected = x_train.iloc[:, selected_indices]
x_test_selected = x_test.iloc[:, selected_indices]

"""fitting selected features into model"""
# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(x_train_selected, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test_selected)


#actual data vs predicted
pred_vs_actu_df = pd.DataFrame({'actual_data':y_test,
                                'predicted_data':y_pred})
print(pred_vs_actu_df.head())



"""=======================================MODEL METRIX===================================================="""

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}")

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Generating confusion matrix
print("confusion_metrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


"""--------plot for Classification report--------"""

# plotting classification report
cr_report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(cr_report).transpose()

# Set up the matplotlib figure
plt.figure(figsize=(10, 10))

# Create a heatmap for the classification report
sns.heatmap(report_df.iloc[:-1, :].drop(columns='support'), annot=True, cmap='Blues', fmt='.2f')

# Set the labels and title
plt.title('Classification Report Heatmap')
plt.xlabel('Metrics', fontsize=15)
plt.ylabel('Classes', fontsize=15)
plt.xticks(rotation=45)
plt.savefig('classification_report.png')


"""-----------plot for confusion matrix---------"""

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_metrix.png')





"""==========================CUSTOMER SEGMENTATION BASED ON USAGE========================================"""

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['total_day_calls', 'total_eve_calls', 'total_night_calls', 'total_intl_calls', 'churn']])

"""inertia plot for clusters"""
plt.figure(figsize=(10, 10))
inertia = []
for n in range(1, 10):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig('elbow_inertia_score.png')

"""silhouette score for clusters"""

plt.figure(figsize=(10, 10))
silhouette_scores = []
for n in range(2, 10):
    kmeans = KMeans(n_clusters=n)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 10), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.savefig('silhouette_score.png')

# Define the number of clusters (this can be fine-tuned later)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add the cluster labels to your original dataframe
df['cluster'] = clusters


# Summarize each cluster
df.drop(['customer_id'], axis=1).groupby('cluster').mean()


# Example using two features for visualization

plt.figure(figsize=(10, 10))
sns.scatterplot(x='total_day_minutes', y='total_intl_minutes', hue='cluster', data=df, palette='Set1')
plt.title('Customer Segments Based on Usage Patterns')
plt.savefig('segments.png')

segment_df = pd.DataFrame({'customer':bf['customer_id'],
                           'segment':clusters})

print(segment_df.head())




"""==================BEHAVIOUR ANALYSIS FOR CHURN BASED ON SERVICE CALLS================================"""
"""churn percentage"""

plt.figure(figsize=(10, 10))
colors = ['#4D3425','#E4512B']
ax = (df['churn'].value_counts()*100.0 /len(df)).plot(kind='bar',
                                                        stacked = True,
                                                        rot = 0,
                                                        color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.text(0.8, 80, "0 - not churned", fontsize=12, color='#4D3425')
plt.text(0.8, 75, " 1 - churned", fontsize=12, color='#E4512B')
ax.set_ylabel('% Customers')
ax.set_xlabel('Churners')
ax.set_ylabel('% Customers')
ax.set_title('churn Distribution')

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-3.5,
            str(round((i.get_height()/total), 1))+'%',
            fontsize=12,
            color='white',
           weight = 'bold')

plt.savefig('churn_percentage.png')

"""number of cust call vs churn"""

plt.figure(figsize=(10, 10))
sns.catplot(x='churn', y='number_customer_service_calls', kind="box", data=df)
plt.savefig('churn_vs_customer_call.png')




"""=======================CUSTOMER LIFETIME VALUE================================================"""


df['total_charges'] = (df['total_day_charge']+df['total_eve_charge']+
                             df['total_night_charge']+df['total_intl_charge'])

df['total_min_usage'] = (df['total_day_minutes'] + df['total_eve_minutes'] +
                     df['total_night_minutes'] + df['total_intl_minutes'])


# ARPU is the monthly revenue in this case #ARPU = average revenue per unit
df['ARPU'] = df['total_charges']/ 4

# Calculate CLV as ARPU * Account Length
df['CLV'] = df['ARPU'] * df['account_length']


# Define revenue per usage minute
usage_value_per_minute = 0.05

# Calculate usage revenue
df['usage_revenue'] = df['total_min_usage'] * usage_value_per_minute

# Update CLV to include usage-based revenue
df['CLV'] += df['usage_revenue']

# Preview the final results
print(df[['customer_id', 'total_charges', 'account_length', 'usage_revenue', 'CLV']])

value_df = pd.DataFrame({'customer_id':df['customer_id'],
                        'account_length':df['account_length'],
                        'clv':df["CLV"],
                         'total_charges':df['total_charges'],
                         'total_min_usage':df['total_min_usage'],
                         'usage_revenue':df['usage_revenue']})
value_sorted = value_df.sort_values(by='clv', ascending=False)
print(value_sorted.head())





"""==============================SERVICE PLAN RELATION TO CHURN=================================="""

# Contingency table for International Plan and Churn
contingency_international = pd.crosstab(df['international_plan'], df['churn'])
print("Contingency Table for International Plan:")
print(contingency_international)

# Contingency table for Voice Mail Plan and Churn
contingency_voice_mail = pd.crosstab(df['voice_mail_plan'], df['churn'])
print("\nContingency Table for Voice Mail Plan:")
print(contingency_voice_mail)


# Chi-Square Test for International Plan
chi2_international, p_international, _, _ = stats.chi2_contingency(contingency_international)
print(f"\nChi-Square Test for International Plan: chi2 = {chi2_international}, p-value = {p_international}")

# Chi-Square Test for Voice Mail Plan
chi2_voice_mail, p_voice_mail, _, _ = stats.chi2_contingency(contingency_voice_mail)
print(f"Chi-Square Test for Voice Mail Plan: chi2 = {chi2_voice_mail}, p-value = {p_voice_mail}")


# Plotting churn rates for International Plan and Voice Mail Pan
plt.figure(figsize=(12, 5))

# Churn rate for International Plan
plt.subplot(1, 2, 1)
sns.barplot(x=contingency_international.index, y=contingency_international[1] / contingency_international.sum(axis=1))
plt.title("Churn Rate by International Plan")
plt.xlabel("International Plan")
plt.ylabel("Churn Rate (Proportion)")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])

# Churn rate for Voice Mail Plan
plt.subplot(1, 2, 2)
sns.barplot(x=contingency_voice_mail.index, y=contingency_voice_mail[1] / contingency_voice_mail.sum(axis=1))
plt.title("Churn Rate by Voice Mail Plan")
plt.xlabel("Voice Mail Plan")
plt.ylabel("Churn Rate (Proportion)")
plt.xticks(ticks=[0, 1], labels=["No", "Yes"])

plt.tight_layout()
plt.savefig('plan_impact_on_churn')




"""============================CALL USAGE PATTERN RELATED TO CHURN==============================="""
"""box plot"""
plt.figure(figsize=(12, 6))

# Total Day Minutes
plt.subplot(1, 3, 1)
sns.boxplot(x='churn', y='total_day_minutes', data=df)
plt.title("Total Day Minutes vs Churn")

# Total Day Charges
plt.subplot(1, 3, 2)
sns.boxplot(x='churn', y='total_day_charge', data=df)
plt.title("Total Day Charge vs Churn")

# Total Evening Minutes
plt.subplot(1, 3, 3)
sns.boxplot(x='churn', y='total_eve_minutes', data=df)
plt.title("Total Evening Minutes vs Churn")

plt.tight_layout()
plt.savefig("pattern.png")


"""violinplot"""

plt.figure(figsize=(12, 6))

# Total Day Minutes
plt.subplot(1, 3, 1)
sns.violinplot(x='churn', y='total_day_minutes', data=df)
plt.title("Total Day Minutes vs Churn")

# Total Day Charges
plt.subplot(1, 3, 2)
sns.violinplot(x='churn', y='total_day_charge', data=df)
plt.title("Total Day Charge vs Churn")

# Total Evening Minutes
plt.subplot(1, 3, 3)
sns.violinplot(x='churn', y='total_eve_minutes', data=df)
plt.title("Total Evening Minutes vs Churn")

plt.tight_layout()
plt.savefig('pattern_type2.png')

"""stats for churn related to usage"""
# T-test for total day minutes
churned_minutes = df[df['churn'] == 1]['total_day_minutes']
non_churned_minutes = df[df['churn'] == 0]['total_day_minutes']

t_stat, p_val = stats.ttest_ind(churned_minutes, non_churned_minutes)
print(f'T-test for Total Day Minutes: t-statistic = {t_stat}, p-value = {p_val}')

