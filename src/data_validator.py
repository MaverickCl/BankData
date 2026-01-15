"""
Validador de Datos
==================
Valida que el dataset tenga las columnas y valores esperados antes del análisis.
"""

import pandas as pd
from typing import List, Optional, Tuple
from src.config import ColumnMapping, ValueMapping


class DataValidationError(Exception):
    """Excepción personalizada para errores de validación."""
    pass


def validate_columns(df: pd.DataFrame, columns: ColumnMapping) -> Tuple[bool, List[str]]:
    """
    Valida que el DataFrame tenga todas las columnas requeridas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar
    columns : ColumnMapping
        Mapeo de columnas esperadas
        
    Returns
    -------
    Tuple[bool, List[str]]
        (True si todas las columnas existen, lista de columnas faltantes)
    """
    required_cols = [
        columns.target, columns.contact, columns.month, columns.day,
        columns.age, columns.balance, columns.duration, columns.campaign,
        columns.housing, columns.loan, columns.job
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    return len(missing_cols) == 0, missing_cols


def validate_target_values(df: pd.DataFrame, columns: ColumnMapping, 
                          values: ValueMapping) -> Tuple[bool, Optional[str]]:
    """
    Valida que la variable objetivo tenga los valores esperados.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar
    columns : ColumnMapping
        Mapeo de columnas
    values : ValueMapping
        Mapeo de valores esperados
        
    Returns
    -------
    Tuple[bool, Optional[str]]
        (True si los valores son válidos, mensaje de error si no)
    """
    if columns.target not in df.columns:
        return False, f"Columna '{columns.target}' no encontrada"
    
    unique_values = df[columns.target].dropna().unique()
    expected_values = {values.target_positive, values.target_negative}
    actual_values = set(unique_values)
    
    if not actual_values.issubset(expected_values):
        unexpected = actual_values - expected_values
        return False, f"Valores inesperados en '{columns.target}': {unexpected}"
    
    return True, None


def validate_numeric_ranges(df: pd.DataFrame, columns: ColumnMapping) -> List[str]:
    """
    Valida que las columnas numéricas estén en rangos razonables.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar
    columns : ColumnMapping
        Mapeo de columnas
        
    Returns
    -------
    List[str]
        Lista de advertencias (vacía si todo está bien)
    """
    warnings = []
    
    # Validar edad
    if columns.age in df.columns:
        if (df[columns.age] < 0).any():
            warnings.append(f"Edades negativas encontradas en '{columns.age}'")
        if (df[columns.age] > 120).any():
            warnings.append(f"Edades mayores a 120 encontradas en '{columns.age}'")
    
    # Validar día del mes
    if columns.day in df.columns:
        if (df[columns.day] < 1).any() or (df[columns.day] > 31).any():
            warnings.append(f"Valores de día fuera del rango 1-31 en '{columns.day}'")
    
    # Validar duración (debe ser positiva)
    if columns.duration in df.columns:
        if (df[columns.duration] < 0).any():
            warnings.append(f"Duraciones negativas encontradas en '{columns.duration}'")
    
    return warnings


def validate_data(df: pd.DataFrame, 
                  columns: ColumnMapping = None,
                  values: ValueMapping = None,
                  strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Valida completamente el dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar
    columns : ColumnMapping, optional
        Mapeo de columnas (usa default si no se proporciona)
    values : ValueMapping, optional
        Mapeo de valores (usa default si no se proporciona)
    strict : bool
        Si True, lanza excepción en errores críticos
        
    Returns
    -------
    Tuple[bool, List[str]]
        (True si válido, lista de advertencias/errores)
    """
    from src.config import DEFAULT_COLUMNS, DEFAULT_VALUES
    
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    
    errors = []
    warnings = []
    
    # Validar columnas
    cols_valid, missing_cols = validate_columns(df, columns)
    if not cols_valid:
        error_msg = f"Columnas faltantes: {missing_cols}"
        if strict:
            raise DataValidationError(error_msg)
        errors.append(error_msg)
    
    # Validar valores de target
    target_valid, target_msg = validate_target_values(df, columns, values)
    if not target_valid:
        if strict:
            raise DataValidationError(target_msg)
        errors.append(target_msg)
    
    # Validar rangos numéricos
    warnings.extend(validate_numeric_ranges(df, columns))
    
    all_valid = len(errors) == 0
    
    return all_valid, errors + warnings
