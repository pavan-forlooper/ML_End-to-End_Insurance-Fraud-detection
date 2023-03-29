# ML_End-to-End_Insurance-Fraud-detection

This is an End to End ML project, to predict vehicle insurance fraud

Insurance Company: https://www.berkshireinsuranceservices.com/arecombinedsinglelimitsbetter

PROBLEM STATEMENT: To find If an insurance claim made is fraud

Steps performed: 
1.	Data ingestion
2.	Data validation 
3.	Data cleaning
4.	Data dividing into clusters.
5.	Train individual models for respective clusters
6.	Save model. 
7.	Prediction – all data pre-processing
8.	Logging
9.	Deployment
10.	Processing client files for prediction


The data contains the following attributes: Features:

1.	months_as_customer: It denotes the number of months for which the customer is associated with the insurance company.
2.	age: continuous. It denotes the age of the person.
3.	policy_number: The policy number.
4.	policy_bind_date: Start date of the policy.
5.	policy_state: The state where the policy is registered.
6.	policy_csl-combined single limits. How much of the bodily injury will be covered from the total damage.
https://www.berkshireinsuranceservices.com/arecombinedsinglelimitsbetter  
7.	policy_deductable: The amount paid out of pocket by the policy-holder before an insurance provider will pay any expenses.
8.	policy_annual_premium: The yearly premium for the policy.
9.	umbrella_limit: An umbrella insurance policy is extra liability insurance coverage that goes beyond the limits of the insured's homeowners, auto or watercraft insurance. It provides an additional layer of security to those who are at risk of being sued for damages to other people's property or injuries caused to others in an accident.
10.	insured_zip: The zip code where the policy is registered.
11.	insured_sex: It denotes the person's gender.
12.	insured_education_level: The highest educational qualification of the policy-holder.
13.	insured_occupation: The occupation of the policy-holder.
14.	insured_hobbies: The hobbies of the policy-holder.
15.	insured_relationship: Dependents on the policy-holder.
16.	capital-gain: It denotes the monitory gains by the person.
17.	capital-loss: It denotes the monitory loss by the person.
18.	incident_date: The date when the incident happened.
19.	incident_type: The type of the incident.
20.	collision_type: The type of collision that took place.
21.	incident_severity: The severity of the incident.
22.	authorities_contacted: Which authority was contacted.
23.	incident_state: The state in which the incident took place.
24.	incident_city: The city in which the incident took place. 
25.	incident_location: The street in which the incident took place.
26.	incident_hour_of_the_day: The time of the day when the incident took place.
27.	property_damage: If any property damage was done.
28.	bodily_injuries: Number of bodily injuries.
29.	Witnesses: Number of witnesses present.
30.	police_report_available: Is the police report available.
31.	total_claim_amount: Total amount claimed by the customer.
32.	injury_claim: Amount claimed for injury
33.	property_claim: Amount claimed for property damage.
34.	vehicle_claim: Amount claimed for vehicle damage.
35.	auto_make: The manufacturer of the vehicle
36.	auto_model: The model of the vehicle. 
37.	auto_year: The year of manufacture of the vehicle. 


Target Label: Whether the claim is fraudulent or not.
38.	fraud_reported:  Y or N



Data flow:
1)	Main.py
a)	Home
b)	Training
i)	Validation
ii)	Actual training
(1)	Pre-processing
(2)	Clustering
(3)	Model training
c)	Prediction 
i)	Validation
ii)	Actual training
(1)	Pre-processing
(2)	Clustering
(3)	Model prediction

