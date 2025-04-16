-- =========================================================================
--  init.sql
--  Script de inicialización de la base de datos 'postsecondary'
-- =========================================================================

-- Opcional: Crear esquema (si se quiere algo distinto de public)
-- CREATE SCHEMA IF NOT EXISTS postsecondary;
-- SET search_path TO postsecondary;

--------------------------------------------------------
-- Tabla: Person
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "Person" (
    "PersonId"           SERIAL PRIMARY KEY,
    "FirstName"          VARCHAR(100) NOT NULL,
    "MiddleName"         VARCHAR(100),
    "LastName"           VARCHAR(100) NOT NULL,
    "DateOfBirth"        DATE,
    "Gender"             VARCHAR(20),
    "EmailAddress"       VARCHAR(255),
    "PhoneNumber"        VARCHAR(50),
    "CreatedOn"          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

--------------------------------------------------------
-- Tabla: OperationPersonRole
-- Relaciona una persona y la(s) función(es)/roles que desempeña.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "OperationPersonRole" (
    "OperationPersonRoleId"  SERIAL PRIMARY KEY,
    "PersonId"               INTEGER NOT NULL,
    "RoleName"               VARCHAR(100) NOT NULL,
    "StartDate"              DATE,
    "EndDate"                DATE,
    CONSTRAINT fk_op_person 
        FOREIGN KEY ("PersonId") 
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: FinancialAid
-- Información base sobre ayudas financieras otorgadas.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "FinancialAid" (
    "FinancialAidId"         SERIAL PRIMARY KEY,
    "PersonId"               INTEGER NOT NULL,
    "ApplicationDate"        DATE,
    "AidType"                VARCHAR(100),  -- Ej: Beca, Préstamo, Subsidio, etc.
    "AmountRequested"        DECIMAL(12,2),
    "AmountGranted"          DECIMAL(12,2),
    "Status"                 VARCHAR(50),   -- Ej: Pendiente, Aprobado, Rechazado
    CONSTRAINT fk_fa_person
        FOREIGN KEY ("PersonId")
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: FinancialAidResultText
-- Mensajes o resultados asociados a la ayuda financiera.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "FinancialAidResultText" (
    "FinancialAidResultTextId"  SERIAL PRIMARY KEY,
    "FinancialAidId"           INTEGER NOT NULL,
    "ResultText"               TEXT,
    "DatePosted"               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_fa_result
        FOREIGN KEY ("FinancialAidId")
        REFERENCES "FinancialAid" ("FinancialAidId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: FinancialAidAward
-- Premios o asignaciones concretas dentro de la ayuda financiera.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "FinancialAidAward" (
    "FinancialAidAwardId"   SERIAL PRIMARY KEY,
    "FinancialAidId"        INTEGER NOT NULL,
    "AwardName"             VARCHAR(100),   -- Ej: "Beca Excelencia"
    "AwardAmount"           DECIMAL(12,2),
    "AwardDate"             DATE,
    CONSTRAINT fk_fa_award
        FOREIGN KEY ("FinancialAidId")
        REFERENCES "FinancialAid" ("FinancialAidId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: PsStudentAcademicRecord
-- Registro académico de un estudiante (calificaciones globales, historial).
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentAcademicRecord" (
    "StudentAcademicRecordId" SERIAL PRIMARY KEY,
    "PersonId"                INTEGER NOT NULL,
    "GPA"                     DECIMAL(4,2),       -- Promedio general
    "AcademicStanding"        VARCHAR(50),        -- Ej: En buen estado, Probatoria, etc.
    "CreditsEarned"          INTEGER DEFAULT 0,
    "CreditsAttempted"       INTEGER DEFAULT 0,
    CONSTRAINT fk_acadrec_person
        FOREIGN KEY ("PersonId")
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: PsStudentAcademicAward
-- Premios académicos de un estudiante.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentAcademicAward" (
    "StudentAcademicAwardId"   SERIAL PRIMARY KEY,
    "StudentAcademicRecordId"  INTEGER NOT NULL,
    "AwardTitle"               VARCHAR(255),
    "AwardDate"                DATE,
    CONSTRAINT fk_acadaward_acadrec
        FOREIGN KEY ("StudentAcademicRecordId")
        REFERENCES "PsStudentAcademicRecord" ("StudentAcademicRecordId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: PsStudentEmergencyContact
-- Contactos de emergencia del estudiante.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentEmergencyContact" (
    "StudentEmergencyContactId"  SERIAL PRIMARY KEY,
    "PersonId"                   INTEGER NOT NULL,
    "ContactName"                VARCHAR(255) NOT NULL,
    "Relationship"               VARCHAR(100),  -- Ej: Madre, Padre, Hermano, etc.
    "PhoneNumber"                VARCHAR(50),
    "EmailAddress"               VARCHAR(255),
    CONSTRAINT fk_emerg_person
        FOREIGN KEY ("PersonId")
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: PsStudentEmployment
-- Información laboral del estudiante (si aplica).
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentEmployment" (
    "StudentEmploymentId"  SERIAL PRIMARY KEY,
    "PersonId"             INTEGER NOT NULL,
    "EmployerName"         VARCHAR(255),
    "JobTitle"             VARCHAR(255),
    "StartDate"            DATE,
    "EndDate"              DATE,
    CONSTRAINT fk_emp_person
        FOREIGN KEY ("PersonId")
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: PsStudentProgram
-- Programas académicos en los que el estudiante está inscrito.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentProgram" (
    "StudentProgramId"   SERIAL PRIMARY KEY,
    "PersonId"           INTEGER NOT NULL,
    "ProgramName"        VARCHAR(255) NOT NULL,
    "Department"         VARCHAR(255),
    "StartTerm"          VARCHAR(50),   -- Ej: "2025-Fall", "2026-Spring"
    "EndTerm"            VARCHAR(50),
    "ProgramStatus"      VARCHAR(50),   -- Ej: "Activo", "Completo", "Suspendido"
    CONSTRAINT fk_prog_person
        FOREIGN KEY ("PersonId")
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: ClassSection
-- Cada sección de clase (ej: curso A, con su horario, etc.).
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "ClassSection" (
    "ClassSectionId"   SERIAL PRIMARY KEY,
    "SectionName"      VARCHAR(255) NOT NULL,
    "CourseCode"       VARCHAR(50) NOT NULL,  -- Ej: "MATH101"
    "InstructorName"   VARCHAR(255),
    "Schedule"         VARCHAR(255),  -- Ej: "Lun, Mie, Vie 8am-9am"
    "Room"             VARCHAR(50)
);

--------------------------------------------------------
-- Tabla: PsStudentEnrollment
-- Matriculación del estudiante en la institución (datos generales).
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentEnrollment" (
    "StudentEnrollmentId"  SERIAL PRIMARY KEY,
    "PersonId"            INTEGER NOT NULL,
    "EnrollmentDate"      DATE,
    "EnrollmentStatus"    VARCHAR(50),    -- Ej: "Activo", "Inactivo", "Graduado", etc.
    "ProgramId"           INTEGER,        -- Referencia a un program (opcional)
    CONSTRAINT fk_enroll_person
        FOREIGN KEY ("PersonId")
        REFERENCES "Person" ("PersonId")
        ON DELETE CASCADE
);

--------------------------------------------------------
-- Tabla: PsStudentClassSection
-- Relación de estudiante (matrícula) con una sección de clase.
--------------------------------------------------------
CREATE TABLE IF NOT EXISTS "PsStudentClassSection" (
    "StudentClassSectionId"   SERIAL PRIMARY KEY,
    "StudentEnrollmentId"     INTEGER NOT NULL,
    "ClassSectionId"          INTEGER NOT NULL,
    "EnrollmentStatus"        VARCHAR(50),   -- Ej: "Inscrito", "Retirado", "Aprobado"
    "Grade"                   VARCHAR(10),   -- Ej: "A", "B", "C", etc.
    CONSTRAINT fk_stsec_enroll
        FOREIGN KEY ("StudentEnrollmentId")
        REFERENCES "PsStudentEnrollment" ("StudentEnrollmentId")
        ON DELETE CASCADE,
    CONSTRAINT fk_stsec_class
        FOREIGN KEY ("ClassSectionId")
        REFERENCES "ClassSection" ("ClassSectionId")
        ON DELETE CASCADE
);

-- =========================================================================
-- Fin del script init.sql
-- =========================================================================
