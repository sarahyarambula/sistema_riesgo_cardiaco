import os
import sqlite3
import hashlib
import secrets


# --------------------------------------------------
# RUTA DE LA BASE DE DATOS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RUTA_DB = os.path.join(
    BASE_DIR,
    "cardiopredict.db"
)


# --------------------------------------------------
# CONEXIÓN A SQLITE
# --------------------------------------------------

def conectar_db():

    conexion = sqlite3.connect(RUTA_DB)

    # Activar claves foráneas en SQLite
    conexion.execute("PRAGMA foreign_keys = ON")

    return conexion


# --------------------------------------------------
# CREAR TABLAS
# --------------------------------------------------

def crear_tablas():

    conexion = conectar_db()
    cursor = conexion.cursor()

    # Tabla de usuarios
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            correo TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla de evaluaciones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER NOT NULL,
            edad INTEGER,
            sexo TEXT,
            antecedentes_cardiacos TEXT,
            tratamiento_previo TEXT,
            medicamentos TEXT,
            observaciones TEXT,
            resultado TEXT,
            probabilidad REAL,
            fecha_evaluacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (usuario_id)
            REFERENCES usuarios(id)
        )
    """)

    conexion.commit()
    conexion.close()


# --------------------------------------------------
# SEGURIDAD DE CONTRASEÑAS
# --------------------------------------------------

def generar_hash_contrasena(contrasena, salt=None):

    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        contrasena.encode("utf-8"),
        salt.encode("utf-8"),
        200000
    ).hex()

    return password_hash, salt


# --------------------------------------------------
# REGISTRAR USUARIO
# --------------------------------------------------

def registrar_usuario(nombre, correo, contrasena):

    conexion = conectar_db()
    cursor = conexion.cursor()

    # Normalizar datos
    nombre = nombre.strip()
    correo = correo.strip().lower()

    password_hash, password_salt = generar_hash_contrasena(
        contrasena
    )

    try:

        cursor.execute("""
            INSERT INTO usuarios (
                nombre,
                correo,
                password_hash,
                password_salt
            )
            VALUES (?, ?, ?, ?)
        """, (
            nombre,
            correo,
            password_hash,
            password_salt
        ))

        conexion.commit()

        return True, "Usuario registrado correctamente."

    except sqlite3.IntegrityError:

        return False, "El correo ya está registrado."

    finally:

        conexion.close()


# --------------------------------------------------
# VERIFICAR USUARIO
# --------------------------------------------------

def verificar_usuario(correo, contrasena):

    conexion = conectar_db()
    cursor = conexion.cursor()

    correo = correo.strip().lower()

    cursor.execute("""
        SELECT
            id,
            nombre,
            correo,
            password_hash,
            password_salt
        FROM usuarios
        WHERE correo = ?
    """, (correo,))

    usuario = cursor.fetchone()

    conexion.close()

    if usuario is None:
        return None

    (
        usuario_id,
        nombre,
        correo_db,
        hash_guardado,
        salt_guardado
    ) = usuario

    hash_ingresado, _ = generar_hash_contrasena(
        contrasena,
        salt_guardado
    )

    if secrets.compare_digest(
        hash_ingresado,
        hash_guardado
    ):
        return {
            "id": usuario_id,
            "nombre": nombre,
            "correo": correo_db
        }

    return None

# --------------------------------------------------
# GUARDAR EVALUACIÓN
# --------------------------------------------------

def guardar_evaluacion(
    usuario_id,
    edad,
    sexo,
    antecedentes_cardiacos,
    tratamiento_previo,
    medicamentos,
    observaciones,
    resultado,
    probabilidad
):

    conexion = conectar_db()
    cursor = conexion.cursor()

    cursor.execute("""
        INSERT INTO evaluaciones (
            usuario_id,
            edad,
            sexo,
            antecedentes_cardiacos,
            tratamiento_previo,
            medicamentos,
            observaciones,
            resultado,
            probabilidad
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        usuario_id,
        edad,
        sexo,
        antecedentes_cardiacos,
        tratamiento_previo,
        medicamentos,
        observaciones,
        resultado,
        probabilidad
    ))

    conexion.commit()
    conexion.close()


# --------------------------------------------------
# OBTENER EVALUACIONES DE UN USUARIO
# --------------------------------------------------

def obtener_evaluaciones_usuario(usuario_id):

    conexion = conectar_db()
    cursor = conexion.cursor()

    cursor.execute("""
        SELECT
            id,
            edad,
            sexo,
            antecedentes_cardiacos,
            tratamiento_previo,
            medicamentos,
            observaciones,
            resultado,
            probabilidad,
            fecha_evaluacion
        FROM evaluaciones
        WHERE usuario_id = ?
        ORDER BY fecha_evaluacion DESC
    """, (usuario_id,))

    registros = cursor.fetchall()

    conexion.close()

    evaluaciones = []

    for registro in registros:

        evaluaciones.append({
            "ID": registro[0],
            "Edad": registro[1],
            "Sexo": registro[2],
            "Antecedentes cardíacos": registro[3],
            "Tratamiento previo": registro[4],
            "Medicamentos": registro[5],
            "Observaciones": registro[6],
            "Resultado": registro[7],
            "Probabilidad": registro[8],
            "Fecha": registro[9]
        })

    return evaluaciones