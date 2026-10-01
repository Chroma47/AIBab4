# Example 4.7 (Modified)
# Added a second Conv2D + MaxPooling2D + Dropout block before Flatten

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input


# ============================================================
# Original model (1 conv block)
# ============================================================
def build_original():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # Conv block 1
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


# ============================================================
# Modified model (2 conv blocks)
# ============================================================
def build_modified():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # Conv block 1
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Conv block 2  <-- NEW
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model


# ============================================================
# Build and compare
# ============================================================
print("=" * 60)
print("ORIGINAL MODEL (1 conv block)")
print("=" * 60)
original = build_original()
original.summary()

print("\n" + "=" * 60)
print("MODIFIED MODEL (2 conv blocks)")
print("=" * 60)
modified = build_modified()
modified.summary()

# ---- Parameter comparison --------------------------------------
print("\n" + "=" * 60)
print("PARAMETER COMPARISON")
print("=" * 60)
print(f"Original total params : {original.count_params():>10,}")
print(f"Modified total params : {modified.count_params():>10,}")
print(f"Difference            : {modified.count_params() - original.count_params():>+10,}")

# Compile (same as original)
original.compile(loss='categorical_crossentropy',
                 optimizer='adam', metrics=['accuracy'])
modified.compile(loss='categorical_crossentropy',
                 optimizer='adam', metrics=['accuracy'])