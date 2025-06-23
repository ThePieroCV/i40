def validate_regression(features, target):
    """
    Valida que las variables 'features' y 'target' para el problema de regresión
    hayan sido definidas correctamente.
    """
    print("Verificando la variable 'target'...")
    assert isinstance(target, str), "La variable 'target' debe ser un string."
    assert target == "costo_total_transporte", "Valor incorrecto para 'target'. Debe ser 'costo_total_transporte'."
    print("✅ 'target' es correcta.")
    print("-" * 20)

    print("Verificando la lista 'features'...")
    assert isinstance(features, list), "La variable 'features' debe ser una lista."
    assert len(features) > 0, "La lista 'features' no puede estar vacía."

    assert "costo_total_transporte" not in features, \
        "La columna 'costo_total_transporte' no debe estar en 'features', ya que es la variable objetivo."

    columnas_esperadas = [
        'distancia_km', 'peso_carga_kg', 'es_refrigerado',
        'es_material_peligroso', 'antiguedad_camion_anios', 'numero_paradas'
    ]

    assert set(features) == set(columnas_esperadas), \
        "La lista de 'features' no coincide con las columnas esperadas. Revisa el notebook."
    print(f"✅ La lista 'features' contiene {len(features)} columnas correctas.")
    print("-" * 20)
    print("\n🎉 ¡Excelente! Las variables 'features' y 'target' están listas para el siguiente paso.")


def validate_regression_model2(model):
    """
    Valida que el modelo sea una instancia correcta de LinearRegression
    y que tenga el random_state configurado en 42.
    """
    print("Verificando la instancia del modelo de Regresión Lineal...")
    from sklearn.linear_model import LinearRegression

    # 1. Verificar que es una instancia de LinearRegression
    assert isinstance(model, LinearRegression), \
        "El modelo debe ser una instancia de 'LinearRegression'."
    print("✅ Tipo de modelo correcto: LinearRegression.")

    # 2. Verificar que el atributo 'random_state' existe
    # Nota: La clase estándar de sklearn.linear_model.LinearRegression no tiene este parámetro.
    # Se asume una clase personalizada o una versión que lo soporte para este ejercicio.
    assert hasattr(model, 'random_state'), \
        "El modelo LinearRegression no tiene el atributo 'random_state'. " \
        "Asegúrate de que la clase que usas lo soporte y lo hayas incluido como parámetro."
    
    # 3. Verificar que el valor de 'random_state' es 42
    assert model.random_state == 42, \
        f"El 'random_state' debe ser 42, pero se encontró el valor '{model.random_state}'."
    print("✅ 'random_state' configurado correctamente.")

    print("\n🎉 ¡Verificación exitosa! El modelo fue instanciado correctamente.")



def validate_regression_model2(model):
    """
    Valida que el modelo sea una instancia correcta de RandomForestRegressor con los parámetros correctos.
    """
    print("Verificando la instancia del modelo de Random Forest...")
    from sklearn.ensemble import RandomForestRegressor

    assert isinstance(model, RandomForestRegressor), \
        "El modelo debe ser una instancia de 'RandomForestRegressor'."
    print(f"✅ Tipo de modelo correcto: RandomForestRegressor.")

    assert hasattr(model, 'random_state'), \
        "El modelo RandomForestRegressor no tiene el atributo 'random_state'. " \
        "Asegúrate de incluirlo como parámetro."
    assert model.random_state == 42, \
        f"El 'random_state' debe ser 42, pero se encontró el valor '{model.random_state}'."
    print("✅ 'random_state' configurado correctamente.")

    assert hasattr(model, 'n_estimators'), \
        "El modelo RandomForestRegressor no tiene el atributo 'n_estimators'. " \
        "Asegúrate de incluirlo como parámetro."
    assert model.n_estimators == 100, \
        f"El 'n_estimators' debe ser 100, pero se encontró el valor '{model.n_estimators}'."
    print("✅ 'n_estimators' configurado correctamente.")

    print("\n🎉 ¡Verificación exitosa! El modelo fue instanciado correctamente.")