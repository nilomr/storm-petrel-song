import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier
from matplotlib import pyplot as plt
from pykanto.utils.compute import with_pbar
from pykanto.utils.io import load_dataset
from pykanto.utils.paths import ProjDirs
from settings import ROOT_DIR
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──── PROJECT SETUP ──────────────────────────────────────────────────────────

dataset_name = "storm-petrel"
raw_data = ROOT_DIR / "data" / "raw" / dataset_name
DIRS = ProjDirs(ROOT_DIR, raw_data, dataset_name, mkdir=False)

# Load derived dataset
features = pd.read_csv(DIRS.RESOURCES / "features.csv", index_col=0)
# Drop any rows with missing values and print a report
# count rows (not columns) with at least one missing value
print(
    f"Dropped {features.isnull().any(axis=1).sum()} rows with missing values."
)
features.dropna(inplace=True)

# Train model

subspecies = False
conf_mat, feature_importances, report, samples = [], [], [], []
min_sample = 20

for i in with_pbar(range(100)):
    # Balance classes
    features_equalclass = features.groupby("ID").filter(
        lambda x: len(x) > min_sample
    )
    features_equalclass = (
        features_equalclass.groupby("ID")
        .apply(
            lambda x: x.sample(
                min(features_equalclass.ID.value_counts()), replace=True
            )
        )
        .droplevel("ID")
    )

    if subspecies:
        features_equalclass = (
            features_equalclass.groupby("group")
            .apply(
                lambda x: x.sample(
                    min(features_equalclass.group.value_counts(), replace=True)
                )
            )
            .droplevel("group")
        )
        y = features_equalclass.group.values
    else:
        y = features_equalclass.ID.values

    samples.append(features_equalclass.index.values)
    X = features_equalclass.drop(["ID", "group"], axis=1)
    strata = features_equalclass.ID.values

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i, stratify=strata
    )
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=X.columns)

    # Train and fit Random Forest
    randforest = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=i,
        class_weight="balanced_subsample",
        sampling_strategy="all",
        max_features="sqrt",
        bootstrap=True,
        replacement=True,
    )
    randforest.fit(X_train, y_train)
    y_pred = randforest.predict(X_test)

    # Collect metrics
    conf_mat.append(confusion_matrix(y_test, y_pred, normalize="all"))
    feature_importances.append(randforest.feature_importances_)
    report.append(
        classification_report(
            y_test, y_pred, target_names=np.unique(y_test), output_dict=True
        )
    )

report_df = pd.concat([pd.DataFrame(x) for x in report], axis=0).reset_index(
    level=0
)


# %%
# Plot confusion matrix

figsize = (7, 7)

conf_mat = np.array(conf_mat)
mean_conf_mat = np.sum(conf_mat, axis=0) / conf_mat.shape[0]
mean_conf_mat = (
    mean_conf_mat.astype("float") / mean_conf_mat.sum(axis=1)[:, np.newaxis]
)

if subspecies:
    labels = [f"H. p. {i}" for i in np.unique(y_test)]
else:
    labels = [x.title().replace("_", " ") for x in np.unique(y_test)]

fig, ax = plt.subplots(figsize=figsize)
sns.set_theme(font_scale=1.4)  # for label size
sns.heatmap(
    data=mean_conf_mat,
    annot=True,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 16},
    square=True,
    fmt=".2f",
    cbar=False,
    cmap="BuPu",
)

ax.set_xlabel("\nPredicted", fontsize=16)
ax.set_ylabel("True label\n", fontsize=16)
plt.xticks(rotation=45, ha="right")

plt.show()


# %%

# Get and save feature importance from RF runs
feature_importances = np.array(feature_importances)
feats = [
    (X_train.columns[i], feature_importances[:, i])
    for i in range(len(X_train.columns))
]
feats_df = pd.DataFrame(feats, columns=["feature", "value"]).explode(
    column="value"
)
feats_df.to_csv(
    DIRS.RESOURCES
    / f"{'sbsp_' if subspecies else ''}rf_feature_importance.csv",
    index=False,
)


# Quick plot of mean feature importance
# Sort the feature importance in descending order
mean_feature_importances = (
    np.sum(feature_importances, axis=0) / feature_importances.shape[0]
)
sorted_indices = np.argsort(mean_feature_importances)[::-1]

fig, ax = plt.subplots(figsize=(19, 5))
plt.title("Feature Importance")
plt.bar(
    range(X_train.shape[1]),
    mean_feature_importances[sorted_indices],
    align="center",
)
plt.xticks(
    range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90
)
plt.tight_layout()
plt.show()

# %%
# Prepare features

include = X_train.columns[sorted_indices][:4]
exclude = [col for col in features.columns if col not in include]


labels = features.group
reduce_features = features.drop(exclude, axis=1)
scaled_features = StandardScaler().fit_transform(reduce_features.values)

# %%
# PCA

pca = PCA(n_components=3)
embedding = pca.fit_transform(scaled_features)

# pca = PCA(n_components=2)
# embedding = pca.fit_transform(scaled_features)

# explained_variance = pca.explained_variance_ratio_


# plot
labels = features.group
coldict = {str(x): i for i, x in enumerate(np.unique(labels))}
colours = [sns.color_palette(palette="Set3")[x] for x in labels.map(coldict)]

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colours,
)
plt.gca().set_aspect("equal", "datalim")


back_colour = "white"
text_colour = "#3d3d3d"
grid_colour = "#f2f2f2"


# Plot
labels = features.ID
if "pelagicus" in np.unique(labels):
    coldict = {"pelagicus": "#196182", "melitensis": "#e67717"}
    colours = np.array([x for x in labels.map(coldict)])
else:
    labels = pd.Series([l.title().replace("_", " ") for l in features.ID])
    coldict = {str(x): i for i, x in enumerate(np.unique(labels))}
    colours = np.array(
        [sns.color_palette(palette="Paired")[x] for x in labels.map(coldict)]
    )


fig, ax = plt.subplots(figsize=(10, 10), facecolor=back_colour)

for group in np.unique(labels):
    ix = np.where(labels == group)
    ax.scatter(
        embedding[:, 0][ix],
        embedding[:, 1][ix],
        c=colours[ix],
        alpha=0.7,
        s=80,
        label=(
            "$\it{H. p. }$" f"$\it{group}$"
            if "pelagicus" in np.unique(labels)
            else group
        ),
    )

ax.legend(
    loc="lower left",
    frameon=False,
    labelcolor=text_colour,
    handletextpad=0.1,
    fontsize=12,
)

ax.set_facecolor(grid_colour)
plt.grid(visible=None)
plt.axis("equal")


plt.xticks([])
plt.yticks([])
ax.set_xlabel("PC1", fontsize=20, labelpad=20, color=text_colour)
ax.set_ylabel("PC2", fontsize=20, labelpad=20, color=text_colour)

plt.show()
