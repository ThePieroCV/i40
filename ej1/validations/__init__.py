def validate_binary(features, target_binary):
    print("Verificando la variable 'target_binary'...")
    assert isinstance(target_binary, str), "La variable 'target_binary' debe ser un string."
    assert target_binary == "falla_inminente", "Valor incorrecto para 'target_binary'."
    print("✅ 'target_binary' es correcta.")
    print("-" * 20)
    print("Verificando la lista 'features'...")
    assert isinstance(features, list), "La variable 'features' debe ser una lista."
    assert len(features) > 0, "La lista 'features' no puede estar vacía."
    columnas_no_features = ["falla_inminente", "tipo_falla"]
    for col in columnas_no_features:
        assert col not in features, f"La columna '{col}' no debe estar en la lista 'features', ya que es una variable objetivo (etiqueta)."
    columnas_numericas_y_de_sensor = [
        'temperatura_motor_c', 'vibracion_mm_s', 'corriente_amperios',
        'horas_operacion', 'presion_aceite_psi', 'eficiencia_energetica_pct',
        'ruido_acustico_db', 'desgaste_rodamiento_micras'
    ]
    assert set(features) == set(columnas_numericas_y_de_sensor), "La lista de 'features' no contiene todas las columnas esperadas. Revisa las columnas numéricas y de sensores del resumen estadístico."
    print(f"✅ La lista 'features' contiene {len(features)} columnas correctas.")
    print("-" * 20)
    print("\n🎉 ¡Excelente! Todas las verificaciones se completaron exitosamente.")
    print("Las variables 'features' y 'target_binary' están listas para el siguiente paso.")

def validate_binary_model(model_binary):
    print("Verificando la instancia del modelo...")
    from sklearn.linear_model import LogisticRegression
    assert isinstance(model_binary, LogisticRegression), \
        "La variable 'model_binary' no es una instancia de LogisticRegression. Revisa la clase que utilizaste."
    print("✅ Tipo de modelo correcto (LogisticRegression).")
    assert hasattr(model_binary, 'random_state'), \
        "El modelo no tiene el atributo 'random_state'. Asegúrate de incluirlo como parámetro al inicializarlo."
    assert model_binary.random_state == 42, \
        f"El 'random_state' debe ser 42, pero se encontró el valor '{model_binary.random_state}'."
    print("✅ 'random_state' configurado correctamente.")
    print("\n🎉 ¡Verificación exitosa! El modelo fue instanciado de manera correcta y está listo para ser evaluado.")

def validate_multiclass(target_multiclass):
    assert isinstance(target_multiclass, str), "La variable 'target_multiclass' debe ser un string."
    assert target_multiclass == "tipo_falla", "Valor incorrecto para 'target_multiclass'."
    print("✅ 'target_multiclass' es correcta.")

def validate_multiclass_model(model_multiclass):
    print("Verificando la instancia del modelo...")
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(model_multiclass, RandomForestClassifier), \
        "La variable 'model_multiclass' no es una instancia de RandomForestClassifier. Revisa la clase que utilizaste."
    print("✅ Tipo de modelo correcto (RandomForestClassifier).")
    assert hasattr(model_multiclass, 'random_state'), \
        "El modelo no tiene el atributo 'random_state'. Asegúrate de incluirlo como parámetro al inicializarlo."
    assert model_multiclass.random_state == 42, \
        f"El 'random_state' debe ser 42, pero se encontró el valor '{model_multiclass.random_state}'."
    print("✅ 'random_state' configurado correctamente.")
    assert hasattr(model_multiclass, 'n_estimators'), \
        "El modelo no tiene el atributo 'n_estimators'. Asegúrate de incluirlo como parámetro al inicializarlo."
    assert model_multiclass.n_estimators == 100, \
        f"El 'n_estimators' debe ser 100, pero se encontró el valor '{model_multiclass.n_estimators}'."
    print("✅ 'n_estimators' configurado correctamente.")
    print("\n🎉 ¡Verificación exitosa! El modelo fue instanciado de manera correcta y está listo para ser evaluado.")