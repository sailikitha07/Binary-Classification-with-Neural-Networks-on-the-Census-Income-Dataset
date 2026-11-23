# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset
This project uses a neural network to predict whether a person's income is above or below $50,000 using census data. It processes categorical and numerical features, trains a PyTorch model, and evaluates its performance using loss and accuracy.
### AIM
To develop a binary classification neural network model using the Census Income Dataset to predict whether a person's annual income is above or below $50,000.
### THEORY
The Census Income Dataset contains demographic and employment information about individuals, such as age, sex, education, workclass, occupation, and hours-per-week. The goal is to build a neural network model that classifies a person's income into two categories: <=50K and >50K. The model uses categorical embeddings, batch normalization, ReLU activation, and dropout to improve learning and classification accuracy.
### DESIGN STEPS
#### STEP 1: Import Libraries –
Import PyTorch, NumPy, Pandas, and Scikit-learn libraries.
#### STEP 2: Load Dataset –
Load the Census Income Dataset from the income.csv file.
#### STEP 3: Preprocess Data –
Separate categorical, continuous, and label columns. Encode categorical values and convert the data into tensors. Split the dataset into 25,000 training records and 5,000 testing records.
#### STEP 4: Build Model –
Create a TabularModel neural network using embedding layers for categorical features, batch normalization for continuous features, one hidden layer with 50 neurons, and dropout probability of 0.4.
#### STEP 5: Train Model –
Train the model using CrossEntropyLoss and the Adam optimizer with a learning rate of 0.001 for 300 epochs.
#### STEP 6: Evaluate Model –
Predict the income category using the test dataset and calculate the test loss and classification accuracy.
### PROGRAM
##### Name: Cholimgapuram Sai Likitha
##### Register Number: 212224230046

```

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load Dataset
df = pd.read_csv("income.csv")
df.columns = df.columns.str.strip()

# Remove missing values if any
df = df.replace(["?", ""], np.nan).dropna().reset_index(drop=True)

# 2. Separate columns
cat_cols = ['sex', 'education', 'marital-status',
            'workclass', 'occupation']

cont_cols = ['age', 'education-num', 'hours-per-week']

label_col = 'income'

# Remove duplicate label column
df = df[cat_cols + cont_cols + [label_col]]

# Encode categorical columns
encoders = {}

for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# Encode target
label_encoder = LabelEncoder()
df[label_col] = label_encoder.fit_transform(df[label_col])

# 3. Create arrays and tensors
X_cat = df[cat_cols].values
X_cont = df[cont_cols].values.astype(np.float32)
y = df[label_col].values

X_cat = torch.tensor(X_cat, dtype=torch.long)
X_cont = torch.tensor(X_cont, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 4. Split dataset
if len(df) < 30000:
    raise ValueError("Dataset must contain 30000 valid rows.")

Xc_train, Xc_test, Xn_train, Xn_test, y_train, y_test = train_test_split(
    X_cat, X_cont, y,
    train_size=25000,
    test_size=5000,
    random_state=42,
    stratify=y
)

print("Training records:", len(y_train))
print("Testing records:", len(y_test))


# 5. Define TabularModel
class TabularModel(nn.Module):

    def __init__(self, cat_sizes, num_cont, output_size):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size, min(50, (size + 1) // 2))
            for size in cat_sizes
        ])

        self.bn = nn.BatchNorm1d(num_cont)

        total = sum(e.embedding_dim for e in self.embeddings) + num_cont

        self.fc1 = nn.Linear(total, 50)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x_cat, x_cont):

        x = [emb(x_cat[:, i])
             for i, emb in enumerate(self.embeddings)]

        x.append(self.bn(x_cont))

        x = torch.cat(x, dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 6. Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Create model
cat_sizes = [len(encoders[col].classes_) for col in cat_cols]

model = TabularModel(
    cat_sizes,
    len(cont_cols),
    len(label_encoder.classes_)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# DataLoader
train_dataset = TensorDataset(Xc_train, Xn_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)


# 7. Train model for 300 epochs
for epoch in range(300):

    model.train()
    total_loss = 0

    for xc, xn, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(xc, xn)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print("Epoch:", epoch + 1,
              "Loss:", round(total_loss / len(train_loader), 4))


# 8. Evaluate model
model.eval()

with torch.no_grad():

    outputs = model(Xc_test, Xn_test)

    test_loss = criterion(outputs, y_test).item()

    predictions = torch.argmax(outputs, dim=1)

    accuracy = (predictions == y_test).float().mean().item()

print("\nTest Results")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")


# 9. Bonus: Predict new income
def predict_income(new_data):

    cat_values = []

    for col in cat_cols:
        value = str(new_data[col]).strip()

        if value not in encoders[col].classes_:
            raise ValueError("Unknown category in " + col)

        cat_values.append(encoders[col].transform([value])[0])

    xc = torch.tensor([cat_values], dtype=torch.long)

    xn = torch.tensor(
        [[float(new_data[col]) for col in cont_cols]],
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        output = model(xc, xn)
        result = torch.argmax(output, dim=1).item()

    return label_encoder.inverse_transform([result])[0]


print("\nProgram executed successfully!")
```

### OUTPUT :

<img width="217" height="40" alt="image" src="https://github.com/user-attachments/assets/4ce88a66-1ef9-41dd-a3e7-0c30dcb5deef" />

<img width="263" height="112" alt="image" src="https://github.com/user-attachments/assets/670232d1-001a-4ee9-8cf6-92ce0b97337e" />

<img width="207" height="66" alt="image" src="https://github.com/user-attachments/assets/d0972584-9c73-443b-bbc8-edeaab322af5" />

<img width="262" height="32" alt="image" src="https://github.com/user-attachments/assets/47393e35-65b7-4942-b6f6-8514f76a96e5" />


### RESULT :
The neural network model was successfully trained using the Census Income Dataset to classify income into two categories. The model was evaluated using test loss and accuracy.
