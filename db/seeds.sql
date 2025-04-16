-- =========================================================================
-- seeds.sql
-- Script para poblar la base de datos con datos sintéticos para aproximadamente 100 estudiantes,
-- completando todas las tablas del esquema.
-- =========================================================================

--------------------------------------------------------
-- 1. Insertar datos en la tabla Person
--------------------------------------------------------
-- Se generan 100 registros con nombres, segundo nombre, apellidos, email, fecha de nacimiento,
-- género y número telefónico.
INSERT INTO "Person" ("FirstName", "MiddleName", "LastName", "EmailAddress", "DateOfBirth", "Gender", "PhoneNumber")
SELECT
    first_names[ceil(random() * array_length(first_names, 1))::int] AS "FirstName",
    middle_names[ceil(random() * array_length(middle_names, 1))::int] AS "MiddleName",
    last_names[ceil(random() * array_length(last_names, 1))::int] AS "LastName",
    lower(first_names[ceil(random() * array_length(first_names, 1))::int] || '.' ||
          last_names[ceil(random() * array_length(last_names, 1))::int]) || '@university.edu' AS "EmailAddress",
    CURRENT_DATE - ((20 + (random() * 10)::int) * interval '365 day') AS "DateOfBirth",
    CASE WHEN random() < 0.5 THEN 'Male' ELSE 'Female' END AS "Gender",
    '555-010' || to_char(g, 'FM000') AS "PhoneNumber"
FROM generate_series(1, 1000) g,
(
  SELECT 
    ARRAY['Alice', 'Bob', 'David', 'Diana', 'Eve', 'Jhon', 'Jenny', 'Christian', 'Michelle', 'Daniel'] AS first_names,
    ARRAY['Marie', 'Alexander', 'Lee', 'Ann', 'James', 'Grace', 'Rosa', 'Michael', 'Sophia', 'Lynn'] AS middle_names,
    ARRAY['Ford', 'Smith', 'Lillard', 'Davis', 'Livingston', 'Hernandez', 'Edwards', 'Reynolds', 'Murphy', 'Curry'] AS last_names
) AS names;

--------------------------------------------------------
-- 2. Insertar datos en la tabla OperationPersonRole
--------------------------------------------------------
-- Para cada persona se asigna un rol (Student o Researcher) con fechas de inicio y fin aleatorias.
INSERT INTO "OperationPersonRole" ("PersonId", "RoleName", "StartDate", "EndDate")
SELECT 
  "PersonId",
  CASE WHEN random() < 0.5 THEN 'Student' ELSE 'Researcher' END AS "RoleName",
  CURRENT_DATE - (random() * 730)::int * interval '1 day' AS "StartDate",
  CURRENT_DATE + (random() * 365)::int * interval '1 day' AS "EndDate"
FROM "Person";

--------------------------------------------------------
-- 3. Insertar datos en la tabla FinancialAid
--------------------------------------------------------
-- Se registra una solicitud de ayuda financiera para cada persona.
INSERT INTO "FinancialAid" ("PersonId", "ApplicationDate", "AidType", "AmountRequested", "AmountGranted", "Status")
SELECT
  "PersonId",
  CURRENT_DATE - (random() * 180)::int * interval '1 day' AS "ApplicationDate",
  CASE WHEN random() < 0.5 THEN 'Scholarship' ELSE 'Loan' END AS "AidType",
  round((1000 + random() * 4000)::numeric, 2)::float8 AS "AmountRequested",
  round((1000 + random() * 4000)::numeric, 2)::float8 AS "AmountGranted",
  CASE WHEN random() < 0.7 THEN 'Approved' ELSE 'Pending' END AS "Status"
FROM "Person";

--------------------------------------------------------
-- 4. Insertar datos en la tabla FinancialAidResultText
--------------------------------------------------------
-- Se asocia un mensaje de resultado a cada solicitud de ayuda financiera.
INSERT INTO "FinancialAidResultText" ("FinancialAidId", "ResultText")
SELECT 
  "FinancialAidId", 
  'Financial aid processed for person ' || "PersonId"
FROM "FinancialAid";

--------------------------------------------------------
-- 5. Insertar datos en la tabla FinancialAidAward
--------------------------------------------------------
-- Se asigna un premio (o beneficio) a cada registro de ayuda financiera.
-- Financial aid will only be granted to students who have at least one award
INSERT INTO "FinancialAid" ("PersonId", "ApplicationDate", "AidType", "AmountRequested", "AmountGranted", "Status")
SELECT
    sar."PersonId",
    CURRENT_DATE - (random() * 180)::int * interval '1 day' AS "ApplicationDate",
    CASE WHEN random() < 0.5 THEN 'Scholarship' ELSE 'Loan' END AS "AidType",
    round((1000 + random() * 4000)::numeric, 2)::float8 AS "AmountRequested",
    round((500 + random() * 2000)::numeric, 2)::float8 AS "AmountGranted",
    CASE WHEN random() < 0.7 THEN 'Approved' ELSE 'Pending' END AS "Status"
FROM "PsStudentAcademicAward" saa
JOIN "PsStudentAcademicRecord" sar ON saa."StudentAcademicRecordId" = sar."StudentAcademicRecordId";

--------------------------------------------------------
-- 6. Insertar datos en la tabla PsStudentAcademicRecord
--------------------------------------------------------
-- Se crea un registro académico para cada persona con GPA, estado académico y créditos.
INSERT INTO "PsStudentAcademicRecord" ("PersonId", "GPA", "AcademicStanding", "CreditsEarned", "CreditsAttempted")
SELECT
  "PersonId",
 round((2.0 + random() * 2.0)::numeric, 2) AS "GPA",
  CASE WHEN random() < 0.1 THEN 'Probation' ELSE 'Good' END AS "AcademicStanding",
  (random() * 120)::int AS "CreditsEarned",
  (random() * 130)::int AS "CreditsAttempted"
FROM "Person";

--------------------------------------------------------
-- 7. Insertar datos en la tabla PsStudentAcademicAward
--------------------------------------------------------
-- Se asigna un premio académico a cada registro académico.
INSERT INTO "PsStudentAcademicAward" ("StudentAcademicRecordId", "AwardTitle", "AwardDate")
SELECT 
    sar."StudentAcademicRecordId",
    'Academic Excellence Award',
    CURRENT_DATE - (random() * 60)::int * interval '1 day' AS "AwardDate"
FROM "PsStudentAcademicRecord" sar
WHERE sar."GPA" >= 3.5 AND sar."AcademicStanding" = 'Good';
--------------------------------------------------------
-- 8. Insertar datos en la tabla PsStudentEmergencyContact
--------------------------------------------------------
-- Se agrega un contacto de emergencia para cada persona.
INSERT INTO "PsStudentEmergencyContact" ("PersonId", "ContactName", "Relationship", "PhoneNumber", "EmailAddress")
SELECT 
    "PersonId",
    'Emergency Contact ' || "PersonId",
    'Parent',
    '555-020' || to_char("PersonId", 'FM000'),
    lower('contact' || "PersonId" || '@example.com')
FROM "Person";

--------------------------------------------------------
-- 9. Insertar datos en la tabla PsStudentEmployment
--------------------------------------------------------
-- Se registra un empleo (por ejemplo, internado o part-time) para cada persona.
INSERT INTO "PsStudentEmployment" ("PersonId", "EmployerName", "JobTitle", "StartDate", "EndDate")
SELECT 
    "PersonId",
    'Company ' || (((random() * 10)::int + 1)) AS "EmployerName",
    CASE WHEN random() < 0.5 THEN 'Intern' ELSE 'Part-Time' END AS "JobTitle",
    CURRENT_DATE - (random() * 365)::int * interval '1 day' AS "StartDate",
    CURRENT_DATE + (random() * 365)::int * interval '1 day' AS "EndDate"
FROM "Person";

--------------------------------------------------------
-- 10. Insertar datos en la tabla PsStudentProgram
--------------------------------------------------------
-- Se asigna un programa académico a cada persona, incluyendo término de inicio y de fin.
INSERT INTO "PsStudentProgram" ("PersonId", "ProgramName", "Department", "StartTerm", "EndTerm", "ProgramStatus")
SELECT 
    "PersonId",
    'Program ' || FLOOR(1 + random() * 20)::int,
    'Department ' || (((random() * 5)::int + 1)),
    '2025-Fall',
    '2026-Spring',
    'Active'
FROM "Person";

--------------------------------------------------------
-- 11. Insertar datos en la tabla ClassSection
--------------------------------------------------------
-- Se insertan 10 secciones de clase con información básica.
INSERT INTO "ClassSection" ("SectionName", "CourseCode", "InstructorName", "Schedule", "Room")
SELECT 
   'Section ' || s,
   'COURSE' || LPAD(s::text, 3, '0'),
   'Instructor ' || s,
   'Mon/Wed/Fri 8:00-9:00',
   'Room ' || s
FROM generate_series(1, 10) s;

--------------------------------------------------------
-- 12. Insertar datos en la tabla PsStudentEnrollment
--------------------------------------------------------
-- Se crea una inscripción para cada persona, relacionándola con el programa asignado.
WITH student_enrollments AS (
    SELECT
        p."PersonId",
        CURRENT_DATE - ((random() * 365)::int * interval '1 day') AS "EnrollmentDate",
        CASE WHEN random() < 0.8 THEN 'Active' ELSE 'Inactive' END AS "EnrollmentStatus",
        ROW_NUMBER() OVER (ORDER BY p."PersonId") AS rn
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    WHERE opr."RoleName" = 'Student'
)
INSERT INTO "PsStudentEnrollment" ("PersonId", "EnrollmentDate", "EnrollmentStatus", "ProgramId")
SELECT
    "PersonId",
    "EnrollmentDate",
    "EnrollmentStatus",
    CASE
        WHEN rn <= 20 THEN rn
        ELSE FLOOR(1 + (random()^1.5) * 20)::int
    END AS "ProgramId"
FROM student_enrollments;

--------------------------------------------------------
-- 13. Insertar datos en la tabla PsStudentClassSection
--------------------------------------------------------
-- Para cada inscripción se asignan entre 1 y 3 secciones de clase de forma aleatoria.
INSERT INTO "PsStudentClassSection" ("StudentEnrollmentId", "ClassSectionId", "EnrollmentStatus", "Grade")
SELECT 
  e."StudentEnrollmentId",
  cs."ClassSectionId",
  'Inscrito' AS "EnrollmentStatus",
  CASE 
    WHEN random() < 0.4 THEN 'A' 
    WHEN random() < 0.6 THEN 'B'
    WHEN random() < 0.8 THEN 'C'
    ELSE 'D' 
  END AS "Grade"
FROM "PsStudentEnrollment" e,
LATERAL (
    SELECT cs_sub."ClassSectionId"
    FROM "ClassSection" cs_sub
    ORDER BY random()
    LIMIT (floor(random() * 3) + 1)::int
) AS cs;

-- =========================================================================
-- Fin del script seeds.sql
-- =========================================================================
