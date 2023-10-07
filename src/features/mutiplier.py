from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, FunctionTransformer


def combine_transformer(columns_to_target_encode_and_combine):
    """
    This function returns a custom encoder.
    The custom encoder takes two columns, target-encodes them and merges them into a third column.
    Args:
        columns_to_target_encode_and_combine (pandas Data Frame): Columns to target-encode and combine

    Returns:
        Sklearn Columntransformer: Encoder for Sklearn Pipeline
    """
    combine_transformer = (
        "combine_encode",
        ColumnTransformer(
            transformers=[
                (
                    "target_combine_encode",
                    TargetEncoder(random_state=0, target_type="continuous"),
                    columns_to_target_encode_and_combine,
                ),
                (
                    "combine_encode",
                    FunctionTransformer(
                        func=combine_columns,
                        kw_args={
                            "col1": columns_to_target_encode_and_combine[0],
                            "col2": columns_to_target_encode_and_combine[1],
                        },
                        validate=False,
                    ),
                    columns_to_target_encode_and_combine,
                ),
            ],
            remainder="passthrough",
        ),
        [],
    )

    return combine_transformer


def combine_columns(X, col1, col2):
    """
    This function is used in combine_transformer.
    Args:
        X (pandas DataFrame): df to do the transformation on
        col1 (pandas DataFrame): col1
        col2 (pandas DataFrame): col2

    Returns:
        pandas DataFrame: tranformed pandas DataFrame
    """
    result = X.copy()
    result["combined_column"] = X[col1] * X[col2]
    return result
