def validate_model(model):
    """
    Valida que el modelo sea una instancia correcta de KMeans con los parámetros
    adecuados para el ejercicio de segmentación de clientes.
    """
    print("Verificando la instancia del modelo K-Means...")
    from sklearn.cluster import KMeans

    # 1. Verificar que es una instancia de KMeans
    assert isinstance(model, KMeans), \
        "El modelo debe ser una instancia de 'KMeans'."
    print("✅ Tipo de modelo correcto: KMeans.")

    # 2. Verificar el número de clústeres (k)
    assert hasattr(model, 'n_clusters'), \
        "El modelo no tiene el atributo 'n_clusters'. Asegúrate de incluirlo."
    assert model.n_clusters == 4, \
        f"El número de clústeres ('n_clusters') debe ser 4 según el método del codo, pero se encontró {model.n_clusters}."
    print(f"✅ 'n_clusters' configurado correctamente en {model.n_clusters}.")

    # 3. Verificar el random_state para reproducibilidad
    assert hasattr(model, 'random_state'), \
        "El modelo no tiene el atributo 'random_state'. Asegúrate de incluirlo."
    assert model.random_state == 42, \
        f"El 'random_state' debe ser 42, pero se encontró {model.random_state}."
    print("✅ 'random_state' configurado correctamente.")

    # 4. Verificar el n_init para robustez
    assert hasattr(model, 'n_init'), \
        "El modelo no tiene el atributo 'n_init'. Se recomienda establecerlo en 10."
    assert model.n_init == 10, \
        f"El valor de 'n_init' debe ser 10, pero se encontró {model.n_init}."
    print("✅ 'n_init' configurado correctamente.")

    print("\n🎉 ¡Verificación exitosa! El modelo K-Means fue instanciado de manera correcta.")