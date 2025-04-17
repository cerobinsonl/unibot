-- =========================================================================
-- seeds.sql
-- Script for populating the database with realistic synthetic data for approximately 100 students
-- =========================================================================

-- Arrays for more diverse and realistic data generation
DO $$
DECLARE
    student_rec  RECORD;
    class_rec    RECORD;
    first_names TEXT[] := ARRAY[
        'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'William', 'Isabella', 'James', 'Sophia', 'Oliver',
        'Charlotte', 'Benjamin', 'Amelia', 'Elijah', 'Mia', 'Lucas', 'Harper', 'Mason', 'Evelyn', 'Logan',
        'Abigail', 'Alexander', 'Emily', 'Ethan', 'Elizabeth', 'Jacob', 'Sofia', 'Michael', 'Avery', 'Daniel',
        'Ella', 'Henry', 'Scarlett', 'Jackson', 'Grace', 'Sebastian', 'Chloe', 'Aiden', 'Victoria', 'Matthew',
        'Riley', 'Samuel', 'Aria', 'David', 'Lily', 'Joseph', 'Madison', 'Carter', 'Layla', 'Owen',
        'Zoe', 'Wyatt', 'Penelope', 'John', 'Aubrey', 'Jack', 'Camila', 'Luke', 'Hannah', 'Jayden',
        'Zoey', 'Gabriel', 'Nora', 'Anthony', 'Stella', 'Isaac', 'Leah', 'Grayson', 'Hazel', 'Julian',
        'Ellie', 'Levi', 'Savannah', 'Christopher', 'Audrey', 'Joshua', 'Maya', 'Andrew', 'Claire', 'Lincoln'
    ];
    
    middle_names TEXT[] := ARRAY[
        'Marie', 'Alexander', 'Lee', 'Ann', 'James', 'Grace', 'Rose', 'Michael', 'Elizabeth', 'Lynn',
        'Nicole', 'Thomas', 'Faith', 'William', 'Jean', 'Robert', 'Mae', 'Joseph', 'Katherine', 'John',
        'Louise', 'Daniel', 'Christine', 'Edward', 'Jane', 'Scott', 'Hope', 'Charles', 'Michelle', 'Allen',
        'Frances', 'Ray', 'Victoria', 'Anthony', 'Maria', 'Peter', 'Elise', 'Paul', 'Claire', 'David',
        'Anne', 'Louis', 'Joy', 'Christopher', 'Dawn', 'Ray', 'Belle', 'Francis', 'Ruth', 'George'
    ];
    
    last_names TEXT[] := ARRAY[
        'Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
        'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson',
        'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'Hernandez', 'King',
        'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 'Carter',
        'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins',
        'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey',
        'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray', 'Ramirez',
        'James', 'Watson', 'Brooks', 'Kelly', 'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes', 'Ross'
    ];
    
    email_domains TEXT[] := ARRAY[
        'university.edu', 'u-college.edu', 'students.stateu.edu', 'umail.com', 'campus.net', 'alumni.tech.edu'
    ];
    
    area_codes TEXT[] := ARRAY['201', '202', '212', '213', '301', '302', '303', '305', '310', '312', '404', '408', '415', '505', '510', '512', '602', '617', '702', '713', '802', '830', '901', '916', '919'];
    
    employer_names TEXT[] := ARRAY[
        'TechNova Solutions', 'Metro Healthcare', 'Capital Finance', 'Green Energy Systems', 'Global Logistics',
        'Innovative Research', 'Quantum Computing', 'Sunrise Retail', 'Pacific Northwest Foods', 'Urban Development Corporation',
        'Heritage Construction', 'Digital Media Group', 'Advanced Manufacturing Inc.', 'Stellar Aerospace', 'Modern Education Services',
        'Alpine Outdoors', 'Coastal Hospitality', 'Apex Telecommunications', 'Evergreen Environmental', 'Guardian Security'
    ];
    
    job_titles TEXT[] := ARRAY[
        'Research Assistant', 'Teaching Assistant', 'Intern', 'Junior Developer', 'Lab Assistant', 
        'Student Worker', 'Barista', 'Retail Associate', 'Campus Ambassador', 'Library Assistant',
        'Administrative Assistant', 'Tour Guide', 'Peer Tutor', 'IT Help Desk', 'Resident Advisor',
        'Food Service Worker', 'Fitness Center Staff', 'Marketing Assistant', 'Social Media Coordinator', 'Event Staff'
    ];
    
    program_names TEXT[] := ARRAY[
        'Computer Science', 'Business Administration', 'Mechanical Engineering', 'Psychology', 'Biology',
        'English Literature', 'Economics', 'Chemistry', 'Political Science', 'Mathematics',
        'Sociology', 'Civil Engineering', 'Communications', 'Nursing', 'Physics',
        'History', 'Electrical Engineering', 'Art and Design', 'Education', 'Philosophy'
    ];
    
    departments TEXT[] := ARRAY[
        'School of Engineering', 'Department Business', 'Department of Sciences', 'College of Liberal Arts', 'School of Medicine',
        'School of Law', 'College of Education', 'Department of Mathematics', 'School of Architecture', 'Department of Fine Arts'
    ];
    
    course_subjects TEXT[] := ARRAY[
        'COMP', 'MATH', 'PHYS', 'CHEM', 'BIOL', 'ECON', 'PSYC', 'ENGL', 'HIST', 'POLI',
        'ENGR', 'BUSI', 'ARTS', 'PHIL', 'SOCI', 'COMM', 'SPAN', 'FREN', 'NURS', 'EDUC'
    ];
    
    room_buildings TEXT[] := ARRAY[
        'Science Center', 'Business Building', 'Engineering Hall', 'Arts Complex', 'Medical School',
        'Library', 'Student Center', 'Technology Institute', 'Humanities Building', 'Sports Complex'
    ];
    
    instructors TEXT[] := ARRAY[
        'Dr. Anderson', 'Prof. Wilson', 'Dr. Martinez', 'Dr. Taylor', 'Prof. Harris',
        'Dr. Thompson', 'Prof. Clark', 'Dr. Rodriguez', 'Dr. White', 'Prof. Walker',
        'Dr. Johnson', 'Prof. Lewis', 'Dr. Wright', 'Dr. Scott', 'Prof. Adams'
    ];
    
    course_numbers TEXT[] := ARRAY[
        '101', '102', '201', '202', '301', '302', '401', '402', '110', '120', 
        '210', '220', '310', '320', '410', '420', '150', '250', '350', '450'
    ];
    
    course_names TEXT[] := ARRAY[
        'Introduction to', 'Fundamentals of', 'Advanced', 'Principles of', 'Topics in',
        'Seminar in', 'Laboratory', 'Research Methods for', 'Theory of', 'Applications in',
        'Contemporary Issues in', 'Analysis of', 'Concepts in', 'Perspectives on', 'Studies in'
    ];
    
    contact_relationships TEXT[] := ARRAY[
        'Parent', 'Guardian', 'Sibling', 'Spouse', 'Relative', 'Friend', 'Partner', 'Roommate'
    ];
    
    academic_standings TEXT[] := ARRAY[
        'Good Standing', 'Dean''s List', 'Academic Probation', 'Warning', 'Honors'
    ];
    
    aid_types TEXT[] := ARRAY[
        'Merit Scholarship', 'Need-based Grant', 'Athletic Scholarship', 'Federal Loan', 'Work-Study',
        'Institutional Scholarship', 'Private Loan', 'Research Grant', 'Teaching Assistantship', 'Fellowship'
    ];
    
    aid_statuses TEXT[] := ARRAY['Approved', 'Pending', 'Denied', 'Under Review', 'Conditional'];
    
    award_titles TEXT[] := ARRAY[
        'Academic Excellence Award', 'Dean''s Scholarship', 'Leadership Recognition', 'Research Achievement',
        'Community Service Award', 'Department Honors', 'Outstanding Student Award', 'Merit Scholarship',
        'President''s List', 'Innovation Award', 'Global Studies Recognition', 'Athletic Achievement'
    ];
    
    schedule_patterns TEXT[] := ARRAY[
        'Mon/Wed/Fri 8:00-9:15 AM', 'Tue/Thu 9:30-10:45 AM', 'Mon/Wed 11:00-12:15 PM', 'Tue/Thu 1:00-2:15 PM',
        'Mon/Wed/Fri 2:30-3:20 PM', 'Tue/Thu 3:30-4:45 PM', 'Mon 6:00-8:45 PM', 'Wed 6:00-8:45 PM',
        'Tue 6:00-8:45 PM', 'Thu 6:00-8:45 PM', 'Fri 9:00-11:45 AM', 'Mon/Wed 4:00-5:15 PM'
    ];
    
    class_statuses TEXT[] := ARRAY['Enrolled', 'Waitlisted', 'Withdrawn', 'Completed', 'In Progress'];
    
    grades TEXT[] := ARRAY['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F', 'W', 'P', 'I'];
    
    grade_weights NUMERIC[] := ARRAY[0.25, 0.10, 0.10, 0.20, 0.10, 0.05, 0.10, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005];
    
    gpa NUMERIC;
    result_text TEXT;
    start_year INT;
    end_year INT;
    term TEXT;
    num_students INT := 120; -- Slightly more than 100 for natural attrition
    current_term TEXT;
    course_disciplines TEXT[] := ARRAY['Computer Science', 'Mathematics', 'Physics', 'Biology', 'Economics', 'Psychology', 'Literature'];
    
BEGIN
    -- Set current term context
    current_term := '2025-Spring';
    
    -- Clear existing data if needed (in reverse order of dependencies)
    TRUNCATE TABLE "PsStudentClassSection" CASCADE;
    TRUNCATE TABLE "ClassSection" CASCADE;
    TRUNCATE TABLE "PsStudentEnrollment" CASCADE;
    TRUNCATE TABLE "PsStudentProgram" CASCADE;
    TRUNCATE TABLE "PsStudentEmployment" CASCADE;
    TRUNCATE TABLE "PsStudentEmergencyContact" CASCADE;
    TRUNCATE TABLE "PsStudentAcademicAward" CASCADE;
    TRUNCATE TABLE "PsStudentAcademicRecord" CASCADE;
    TRUNCATE TABLE "FinancialAidAward" CASCADE;
    TRUNCATE TABLE "FinancialAidResultText" CASCADE;
    TRUNCATE TABLE "FinancialAid" CASCADE;
    TRUNCATE TABLE "OperationPersonRole" CASCADE;
    TRUNCATE TABLE "Person" CASCADE;

    -- 1. Insert Person data
    FOR i IN 1..num_students LOOP
        INSERT INTO "Person" (
            "FirstName", 
            "MiddleName", 
            "LastName", 
            "DateOfBirth", 
            "Gender", 
            "EmailAddress", 
            "PhoneNumber"
        )
        VALUES (
            first_names[1 + floor(random()::numeric* array_length(first_names, 1))::int],
            CASE WHEN random()::numeric< 0.7 THEN middle_names[1 + floor(random()::numeric* array_length(middle_names, 1))::int] ELSE NULL END,
            last_names[1 + floor(random()::numeric* array_length(last_names, 1))::int],
            (CURRENT_DATE - ((18 + floor(random()::numeric* 12)::int) * interval '1 year') 
                         - (floor(random()::numeric* 365)::int * interval '1 day')),
            CASE 
                WHEN random()::numeric< 0.49 THEN 'Male'
                WHEN random()::numeric< 0.98 THEN 'Female'
                ELSE 'Non-binary'
            END,
            -- Create email with consistent format based on name
            lower(
                substring(first_names[1 + floor(random()::numeric* array_length(first_names, 1))::int] from 1 for 1) || 
                last_names[1 + floor(random()::numeric* array_length(last_names, 1))::int] || 
                floor(random()::numeric* 999)::text || 
                '@' || email_domains[1 + floor(random()::numeric* array_length(email_domains, 1))::int]
            ),
            -- More realistic phone number format
            area_codes[1 + floor(random()::numeric* array_length(area_codes, 1))::int] || '-' || 
            lpad(floor(random()::numeric* 1000)::text, 3, '0') || '-' || 
            lpad(floor(random()::numeric* 10000)::text, 4, '0')
        );
    END LOOP;

    -- 2. Add roles to all persons
    -- Students (90%)
    INSERT INTO "OperationPersonRole" ("PersonId", "RoleName", "StartDate", "EndDate")
    SELECT 
        "PersonId",
        'Student',
        CURRENT_DATE - (floor(random()::numeric* 1095)::int) * interval '1 day', -- Up to 3 years ago
        CASE 
            WHEN random()::numeric< 0.8 THEN NULL -- Most still active
            ELSE CURRENT_DATE + (floor(random()::numeric* 730)::int) * interval '1 day' -- Expected graduation
        END
    FROM "Person"
    WHERE random()::numeric< 0.9; -- 90% are students

    -- Researchers (some overlap with students - graduate students)
    INSERT INTO "OperationPersonRole" ("PersonId", "RoleName", "StartDate", "EndDate")
    SELECT 
        "PersonId",
        'Researcher',
        CURRENT_DATE - (floor(random()::numeric* 730)::int) * interval '1 day', -- Up to 2 years ago
        CASE 
            WHEN random()::numeric< 0.7 THEN NULL -- Most still active
            ELSE CURRENT_DATE + (floor(random()::numeric* 365)::int) * interval '1 day' -- Expected end date
        END
    FROM "Person"
    WHERE random()::numeric< 0.2; -- 20% are researchers (some overlap with students)
    
    -- Teaching Assistants
    INSERT INTO "OperationPersonRole" ("PersonId", "RoleName", "StartDate", "EndDate")
    SELECT 
        "PersonId",
        'Teaching Assistant',
        CURRENT_DATE - (floor(random()::numeric* 365)::int) * interval '1 day', -- Up to 1 year ago
        CURRENT_DATE + (floor(random()::numeric* 180)::int) * interval '1 day' -- Semester-based end date
    FROM "Person"
    WHERE random()::numeric< 0.15; -- 15% are TAs

    -- 3. Create academic records for all students
    INSERT INTO "PsStudentAcademicRecord" (
        "PersonId", 
        "GPA", 
        "AcademicStanding", 
        "CreditsEarned", 
        "CreditsAttempted"
    )
    SELECT 
        p."PersonId",
        CASE 
            WHEN random()::numeric< 0.05 THEN round((1.5 + random()::numeric* 1.0)::numeric, 2) -- 5% struggling (1.5-2.5)
            WHEN random()::numeric< 0.20 THEN round((2.5 + random()::numeric* 0.5)::numeric, 2) -- 20% below average (2.5-3.0)
            WHEN random()::numeric< 0.50 THEN round((3.0 + random()::numeric* 0.5)::numeric, 2) -- 50% average (3.0-3.5)
            ELSE round((3.5 + random()::numeric* 0.5)::numeric, 2) -- 25% excellent (3.5-4.0)
        END AS "GPA",
        (SELECT academic_standings[1 + floor(random()::numeric* array_length(academic_standings, 1))::int]),
        floor(random()::numeric* 120)::int, -- Credits earned
        floor(random()::numeric* 130)::int  -- Credits attempted (sometimes more than earned)
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    WHERE opr."RoleName" = 'Student'
    GROUP BY p."PersonId"; -- Avoid duplicates if student has multiple roles

    -- Update academic standing based on GPA for realism
    UPDATE "PsStudentAcademicRecord"
    SET "AcademicStanding" = 
        CASE
            WHEN "GPA" < 2.0 THEN 'Academic Probation'
            WHEN "GPA" >= 3.7 THEN 'Dean''s List'
            WHEN "GPA" >= 3.5 THEN 'Honors'
            ELSE 'Good Standing'
        END;
    
    -- Make credits attempted at least as much as credits earned
    UPDATE "PsStudentAcademicRecord" 
    SET "CreditsAttempted" = "CreditsEarned" + floor(random()::numeric* 10)::int
    WHERE "CreditsAttempted" < "CreditsEarned";

    -- 4. Add academic awards (more likely for high GPA students)
    INSERT INTO "PsStudentAcademicAward" (
        "StudentAcademicRecordId", 
        "AwardTitle", 
        "AwardDate"
    )
    SELECT 
        sar."StudentAcademicRecordId",
        (SELECT award_titles[1 + floor(random()::numeric* array_length(award_titles, 1))::int]),
        CURRENT_DATE - (floor(random()::numeric* 365)::int) * interval '1 day'
    FROM "PsStudentAcademicRecord" sar
    WHERE (sar."GPA" >= 3.5 AND random()::numeric< 0.9) -- 90% of high GPA students get awards
       OR (sar."GPA" >= 3.0 AND random()::numeric< 0.3) -- 30% of good GPA students get awards
       OR (sar."GPA" < 3.0 AND random()::numeric< 0.05); -- 5% of lower GPA students get awards

    -- Add multiple awards for some top students
    INSERT INTO "PsStudentAcademicAward" (
        "StudentAcademicRecordId", 
        "AwardTitle", 
        "AwardDate"
    )
    SELECT 
        sar."StudentAcademicRecordId",
        (SELECT award_titles[1 + floor(random()::numeric* array_length(award_titles, 1))::int]),
        CURRENT_DATE - (floor(random()::numeric* 730)::int) * interval '1 day'
    FROM "PsStudentAcademicRecord" sar
    WHERE sar."GPA" >= 3.8 AND random()::numeric< 0.7; -- 70% of top students get a second award

    -- 5. Create financial aid applications
    INSERT INTO "FinancialAid" (
        "PersonId", 
        "ApplicationDate", 
        "AidType", 
        "AmountRequested", 
        "AmountGranted", 
        "Status"
    )
    SELECT
        p."PersonId",
        CURRENT_DATE - (floor(random()::numeric* 180)::int) * interval '1 day',
        (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]),
        -- Amount requested varies by aid type (captured in the variable name)
        CASE 
            WHEN (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]) LIKE '%Scholarship%' THEN round((5000 + random()::numeric* 15000)::numeric, 2)
            WHEN (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]) LIKE '%Grant%' THEN round((2000 + random()::numeric* 8000)::numeric, 2)
            WHEN (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]) LIKE '%Loan%' THEN round((10000 + random()::numeric* 30000)::numeric, 2)
            ELSE round((3000 + random()::numeric* 12000)::numeric, 2)
        END,
        -- Amount granted is typically less than or equal to requested
        CASE 
            WHEN random()::numeric< 0.7 THEN round((0.7 + random()::numeric* 0.3) * 
                CASE 
                    WHEN (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]) LIKE '%Scholarship%' THEN round((5000 + random()::numeric* 15000)::numeric, 2)
                    WHEN (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]) LIKE '%Grant%' THEN round((2000 + random()::numeric* 8000)::numeric, 2)
                    WHEN (SELECT aid_types[1 + floor(random()::numeric* array_length(aid_types, 1))::int]) LIKE '%Loan%' THEN round((10000 + random()::numeric* 30000)::numeric, 2)
                    ELSE round((3000 + random()::numeric* 12000)::numeric, 2)
                END::numeric, 2)
            ELSE NULL -- For non-approved applications
        END,
        (SELECT aid_statuses[1 + floor(random()::numeric* array_length(aid_statuses, 1))::int])
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    WHERE opr."RoleName" = 'Student' AND random()::numeric< 0.8; -- 80% of students apply for aid

    -- Ensure amounts make sense based on status
    UPDATE "FinancialAid"
    SET "AmountGranted" = NULL
    WHERE "Status" IN ('Pending', 'Denied', 'Under Review');

    UPDATE "FinancialAid"
    SET "Status" = 'Approved'
    WHERE "AmountGranted" IS NOT NULL AND "Status" NOT IN ('Approved', 'Conditional');

    -- 6. Add result text to financial aid applications
    INSERT INTO "FinancialAidResultText" (
        "FinancialAidId", 
        "ResultText", 
        "DatePosted"
    )
    SELECT
        fa."FinancialAidId",
        CASE fa."Status"
            WHEN 'Approved' THEN 'Congratulations! Your application for ' || fa."AidType" || ' has been approved for $' || fa."AmountGranted" || '. Please check your student portal for disbursement details.'
            WHEN 'Pending' THEN 'Your application for ' || fa."AidType" || ' is currently under review. Please allow 2-3 weeks for processing.'
            WHEN 'Denied' THEN 'We regret to inform you that your application for ' || fa."AidType" || ' was not approved at this time. Please contact the financial aid office for more information.'
            WHEN 'Under Review' THEN 'Your application requires additional documentation. Please submit the requested information within 14 days.'
            WHEN 'Conditional' THEN 'Your ' || fa."AidType" || ' has been conditionally approved for $' || fa."AmountGranted" || ' pending verification of your enrollment status.'
            ELSE 'Thank you for your ' || fa."AidType" || ' application. Our team will review your submission.'
        END,
        fa."ApplicationDate" + (floor(random()::numeric* 14) + 3)::int * interval '1 day' -- Result 3-17 days after application
    FROM "FinancialAid" fa;

    -- 7. Create financial aid awards for approved applications
    INSERT INTO "FinancialAidAward" (
        "FinancialAidId", 
        "AwardName", 
        "AwardAmount", 
        "AwardDate"
    )
    SELECT
        fa."FinancialAidId",
        CASE 
            WHEN fa."AidType" LIKE '%Scholarship%' THEN 
                CASE WHEN random()::numeric< 0.3 THEN 'Presidential Scholarship'
                     WHEN random()::numeric< 0.6 THEN 'Merit-Based Scholarship'
                     ELSE 'Department Scholarship'
                END
            WHEN fa."AidType" LIKE '%Grant%' THEN 
                CASE WHEN random()::numeric< 0.4 THEN 'Need-Based Grant'
                     WHEN random()::numeric< 0.7 THEN 'University Grant'
                     ELSE 'Diversity Grant'
                END
            WHEN fa."AidType" LIKE '%Loan%' THEN 
                CASE WHEN random()::numeric< 0.5 THEN 'Subsidized Student Loan'
                     ELSE 'Unsubsidized Student Loan'
                END
            ELSE fa."AidType" || ' Award'
        END,
        -- Split amount into 1-3 awards
        CASE 
            WHEN random()::numeric< 0.6 THEN fa."AmountGranted" -- 60% get full amount in one award
            ELSE round((random()::numeric* 0.7 + 0.3) * fa."AmountGranted", 2) -- 40% get partial in multiple awards
        END,
        fa."ApplicationDate" + (floor(random()::numeric* 21) + 7)::int * interval '1 day' -- Award 1-4 weeks after application
    FROM "FinancialAid" fa
    WHERE fa."Status" IN ('Approved', 'Conditional') AND fa."AmountGranted" IS NOT NULL;

    -- Add second awards for some students
    INSERT INTO "FinancialAidAward" (
        "FinancialAidId", 
        "AwardName", 
        "AwardAmount", 
        "AwardDate"
    )
    SELECT
        fa."FinancialAidId",
        CASE 
            WHEN fa."AidType" LIKE '%Scholarship%' THEN 'Supplemental Scholarship'
            WHEN fa."AidType" LIKE '%Grant%' THEN 'Additional Grant Funding'
            WHEN fa."AidType" LIKE '%Loan%' THEN 'Extended Loan'
            ELSE 'Supplemental ' || fa."AidType"
        END,
        -- Remaining amount from initial award
        (SELECT fa."AmountGranted" - COALESCE(SUM(faa."AwardAmount"), 0) 
         FROM "FinancialAidAward" faa 
         WHERE faa."FinancialAidId" = fa."FinancialAidId"),
        fa."ApplicationDate" + (floor(random()::numeric* 14) + 10)::int * interval '1 day' -- Later than first award
    FROM "FinancialAid" fa
    JOIN "FinancialAidAward" faa ON fa."FinancialAidId" = faa."FinancialAidId"
    WHERE fa."Status" IN ('Approved', 'Conditional') 
      AND fa."AmountGranted" IS NOT NULL
      AND random()::numeric< 0.4 -- 40% get split awards
    GROUP BY fa."FinancialAidId", fa."AidType", fa."AmountGranted", fa."ApplicationDate"
    HAVING fa."AmountGranted" > COALESCE(SUM(faa."AwardAmount"), 0);

    -- 8. Add emergency contacts (1-2 per student)
    INSERT INTO "PsStudentEmergencyContact" (
        "PersonId", 
        "ContactName", 
        "Relationship", 
        "PhoneNumber", 
        "EmailAddress"
    )
    SELECT
        p."PersonId",
        -- Generate a plausible contact name
        first_names[1 + floor(random()::numeric* array_length(first_names, 1))::int] || ' ' || 
        last_names[1 + floor(random()::numeric* array_length(last_names, 1))::int],
        (SELECT contact_relationships[1 + floor(random()::numeric* array_length(contact_relationships, 1))::int]),
        -- Phone number
        area_codes[1 + floor(random()::numeric* array_length(area_codes, 1))::int] || '-' || 
        lpad(floor(random()::numeric* 1000)::text, 3, '0') || '-' || 
        lpad(floor(random()::numeric* 10000)::text, 4, '0'),
        -- Email
        lower(
            first_names[1 + floor(random()::numeric* array_length(first_names, 1))::int] || '.' || 
            last_names[1 + floor(random()::numeric* array_length(last_names, 1))::int] || 
            floor(random()::numeric* 99)::text || 
            '@' || 
            CASE WHEN random()::numeric< 0.7 THEN 'gmail.com' ELSE 
                CASE WHEN random()::numeric< 0.5 THEN 'outlook.com' ELSE 'yahoo.com' END
            END
        )
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    WHERE opr."RoleName" = 'Student';

    -- Add second emergency contact for ~60% of students
    INSERT INTO "PsStudentEmergencyContact" (
        "PersonId", 
        "ContactName", 
        "Relationship", 
        "PhoneNumber", 
        "EmailAddress"
    )
    SELECT
        p."PersonId",
        first_names[1 + floor(random()::numeric* array_length(first_names, 1))::int] || ' ' ||
        -- Continuing from the previous code block...
        
        last_names[1 + floor(random()::numeric* array_length(last_names, 1))::int],
        (SELECT r FROM (
            SELECT unnest(contact_relationships) AS r 
            EXCEPT 
            SELECT "Relationship" FROM "PsStudentEmergencyContact" WHERE "PersonId" = p."PersonId"
        ) sub ORDER BY random()::numeric LIMIT 1), -- Different relationship than first contact
        area_codes[1 + floor(random()::numeric* array_length(area_codes, 1))::int] || '-' || 
        lpad(floor(random()::numeric* 1000)::text, 3, '0') || '-' || 
        lpad(floor(random()::numeric* 10000)::text, 4, '0'),
        lower(
            first_names[1 + floor(random()::numeric* array_length(first_names, 1))::int] || '.' || 
            last_names[1 + floor(random()::numeric* array_length(last_names, 1))::int] || 
            floor(random()::numeric* 99)::text || 
            '@' || 
            CASE WHEN random()::numeric< 0.7 THEN 'gmail.com' ELSE 
                CASE WHEN random()::numeric< 0.5 THEN 'outlook.com' ELSE 'yahoo.com' END
            END
        )
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    WHERE opr."RoleName" = 'Student'
    AND random()::numeric< 0.6; -- 60% get a second contact

    -- 9. Add student employment records (60% of students have jobs)
    INSERT INTO "PsStudentEmployment" (
        "PersonId", 
        "EmployerName", 
        "JobTitle", 
        "StartDate", 
        "EndDate"
    )
    SELECT
        p."PersonId",
        (SELECT employer_names[1 + floor(random()::numeric* array_length(employer_names, 1))::int]),
        (SELECT job_titles[1 + floor(random()::numeric* array_length(job_titles, 1))::int]),
        CURRENT_DATE - (floor(random()::numeric* 730) + 30)::int * interval '1 day', -- Started 1 month to 2 years ago
        CASE 
            WHEN random()::numeric< 0.8 THEN NULL -- 80% still employed
            ELSE CURRENT_DATE - floor(random()::numeric* 30)::int * interval '1 day' -- Ended recently
        END
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    WHERE opr."RoleName" = 'Student' AND random()::numeric< 0.6; -- 60% have jobs

    -- Add a second job for some students (past job)
    INSERT INTO "PsStudentEmployment" (
        "PersonId", 
        "EmployerName", 
        "JobTitle", 
        "StartDate", 
        "EndDate"
    )
    SELECT
        p."PersonId",
        e2."EmployerName",
        (SELECT job_titles[1 + floor(random()::numeric* array_length(job_titles, 1))::int]),
        CURRENT_DATE - (floor(random()::numeric* 1095) + 365)::int * interval '1 day', -- Started 1-4 years ago
        CURRENT_DATE - (floor(random()::numeric* 365) + 30)::int * interval '1 day' -- Ended 1 month to 1 year ago
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    JOIN "PsStudentEmployment" e ON p."PersonId" = e."PersonId"
    CROSS JOIN (SELECT "EmployerName" FROM "PsStudentEmployment" ORDER BY random()::numeric LIMIT 1) e2
    WHERE opr."RoleName" = 'Student' AND random()::numeric< 0.3 -- 30% of those with jobs had a previous job
    AND e."EndDate" IS NULL; -- Only if current job is active

    -- 10. Add student programs (all students are in at least one program)
    INSERT INTO "PsStudentProgram" (
        "PersonId", 
        "ProgramName", 
        "Department", 
        "StartTerm", 
        "EndTerm", 
        "ProgramStatus"
    )
    SELECT
        p."PersonId",
        (SELECT program_names[1 + floor(random()::numeric* array_length(program_names, 1))::int]),
        (SELECT departments[1 + floor(random()::numeric* array_length(departments, 1))::int]),
        -- Realistic terms based on start date
        CASE 
            WHEN EXTRACT(MONTH FROM opr."StartDate") BETWEEN 1 AND 5 THEN 
                EXTRACT(YEAR FROM opr."StartDate")::text || '-Spring'
            WHEN EXTRACT(MONTH FROM opr."StartDate") BETWEEN 6 AND 7 THEN 
                EXTRACT(YEAR FROM opr."StartDate")::text || '-Summer'
            ELSE 
                EXTRACT(YEAR FROM opr."StartDate")::text || '-Fall'
        END,
        -- End term is typically 4 years after start for undergrad
        CASE 
            WHEN EXTRACT(MONTH FROM opr."StartDate") BETWEEN 1 AND 5 THEN 
                (EXTRACT(YEAR FROM opr."StartDate") + 4)::text || '-Spring'
            WHEN EXTRACT(MONTH FROM opr."StartDate") BETWEEN 6 AND 7 THEN 
                (EXTRACT(YEAR FROM opr."StartDate") + 4)::text || '-Summer'
            ELSE 
                (EXTRACT(YEAR FROM opr."StartDate") + 4)::text || '-Fall'
        END,
        CASE
            WHEN opr."EndDate" IS NOT NULL AND opr."EndDate" < CURRENT_DATE THEN 'Completed'
            WHEN sar."AcademicStanding" = 'Academic Probation' THEN 'Probation'
            ELSE 'Active'
        END
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    LEFT JOIN "PsStudentAcademicRecord" sar ON p."PersonId" = sar."PersonId"
    WHERE opr."RoleName" = 'Student'
    GROUP BY p."PersonId", opr."StartDate", opr."EndDate", sar."AcademicStanding";

    -- Some students are in dual degree programs
    INSERT INTO "PsStudentProgram" (
        "PersonId", 
        "ProgramName", 
        "Department", 
        "StartTerm", 
        "EndTerm", 
        "ProgramStatus"
    )
    SELECT
        p."PersonId",
        -- Make sure second program is different from first
        (SELECT prog FROM (
            SELECT unnest(program_names) AS prog 
            EXCEPT 
            SELECT "ProgramName" FROM "PsStudentProgram" WHERE "PersonId" = p."PersonId"
        ) sub ORDER BY random()::numeric LIMIT 1),
        (SELECT departments[1 + floor(random()::numeric* array_length(departments, 1))::int]),
        sp."StartTerm",
        sp."EndTerm",
        sp."ProgramStatus"
    FROM "Person" p
    JOIN "PsStudentProgram" sp ON p."PersonId" = sp."PersonId"
    WHERE random()::numeric< 0.15 -- 15% in dual degree programs
    AND sp."ProgramStatus" = 'Active'; -- Only active students get dual programs

    -- 11. Create class sections
    -- First, create an array of possible class sections
    FOR i IN 1..40 LOOP -- Create 40 sections
        INSERT INTO "ClassSection" (
            "SectionName", 
            "CourseCode", 
            "InstructorName", 
            "Schedule", 
            "Room"
        )
        VALUES (
            (SELECT course_names[1 + floor(random()::numeric* array_length(course_names, 1))::int]) || ' ' ||
            (SELECT course_disciplines[1 + floor(random()::numeric* array_length(course_disciplines, 1))::int]),
            (SELECT course_subjects[1 + floor(random()::numeric* array_length(course_subjects, 1))::int]) || ' ' ||
            (SELECT course_numbers[1 + floor(random()::numeric* array_length(course_numbers, 1))::int]),
            (SELECT instructors[1 + floor(random()::numeric* array_length(instructors, 1))::int]),
            (SELECT schedule_patterns[1 + floor(random()::numeric* array_length(schedule_patterns, 1))::int]),
            (SELECT room_buildings[1 + floor(random()::numeric* array_length(room_buildings, 1))::int]) || ' ' ||
            (100 + floor(random()::numeric* 400))::text
        );
    END LOOP;

    -- 12. Create student enrollments
    INSERT INTO "PsStudentEnrollment" (
        "PersonId", 
        "EnrollmentDate", 
        "EnrollmentStatus", 
        "ProgramId"
    )
    SELECT
        p."PersonId",
        opr."StartDate", -- Enrollment date same as role start date
        CASE 
            WHEN sp."ProgramStatus" = 'Active' THEN 'Active'
            WHEN sp."ProgramStatus" = 'Completed' THEN 'Graduated'
            WHEN sp."ProgramStatus" = 'Probation' THEN 'Probation'
            ELSE 'Inactive'
        END,
        sp."StudentProgramId" -- Link to primary program
    FROM "Person" p
    JOIN "OperationPersonRole" opr ON p."PersonId" = opr."PersonId"
    JOIN "PsStudentProgram" sp ON p."PersonId" = sp."PersonId"
    WHERE opr."RoleName" = 'Student'
    GROUP BY p."PersonId", opr."StartDate", sp."ProgramStatus", sp."StudentProgramId";

    -- 13. Enroll students in classes
    -- Each active student takes 3-5 classes
    FOR student_rec IN (
        SELECT se."StudentEnrollmentId", se."PersonId"
        FROM "PsStudentEnrollment" se
        WHERE se."EnrollmentStatus" = 'Active'
    ) LOOP
        -- Get 3-5 random class sections
        FOR class_rec IN (
            SELECT cs."ClassSectionId"
            FROM "ClassSection" cs
            ORDER BY random()
            LIMIT floor(random()::numeric* 3)::int + 3 -- 3-5 classes
        ) LOOP
            -- Generate a grade distribution that's realistic
            gpa := (SELECT "GPA" FROM "PsStudentAcademicRecord" WHERE "PersonId" = student_rec."PersonId");
            
            -- Better GPA means better chance of good grades
            INSERT INTO "PsStudentClassSection" (
                "StudentEnrollmentId",
                "ClassSectionId",
                "EnrollmentStatus",
                "Grade"
            )
            VALUES (
                student_rec."StudentEnrollmentId",
                class_rec."ClassSectionId",
                (SELECT class_statuses[1 + floor(random()::numeric* array_length(class_statuses, 1))::int]),
                -- More likely to get grades close to their GPA
                CASE 
                    WHEN gpa >= 3.7 AND random()::numeric< 0.7 THEN 'A'
                    WHEN gpa >= 3.3 AND random()::numeric< 0.7 THEN 'A-'
                    WHEN gpa >= 3.0 AND random()::numeric< 0.7 THEN 'B+'
                    WHEN gpa >= 2.7 AND random()::numeric< 0.7 THEN 'B'
                    WHEN gpa >= 2.3 AND random()::numeric< 0.7 THEN 'B-'
                    WHEN gpa >= 2.0 AND random()::numeric< 0.7 THEN 'C+'
                    ELSE (SELECT grades[1 + floor(random()::numeric* array_length(grades, 1))::int])
                END
            );
        END LOOP;
    END LOOP;

    -- Fix grades for special enrollment statuses
    UPDATE "PsStudentClassSection"
    SET "Grade" = NULL
    WHERE "EnrollmentStatus" IN ('Enrolled', 'Waitlisted', 'In Progress');

    UPDATE "PsStudentClassSection"
    SET "Grade" = 'W'
    WHERE "EnrollmentStatus" = 'Withdrawn';

    -- Make enrollment statistics more realistic
    -- Count students' credits properly based on their classes
    WITH credit_count AS (
        SELECT 
            sar."StudentAcademicRecordId",
            COUNT(*) * 3 AS total_credits -- Assuming 3 credits per class
        FROM "PsStudentAcademicRecord" sar
        JOIN "Person" p ON sar."PersonId" = p."PersonId"
        JOIN "PsStudentEnrollment" se ON p."PersonId" = se."PersonId"
        JOIN "PsStudentClassSection" scs ON se."StudentEnrollmentId" = scs."StudentEnrollmentId"
        WHERE scs."Grade" NOT IN ('W', 'I') OR scs."Grade" IS NULL
        GROUP BY sar."StudentAcademicRecordId"
    )
    UPDATE "PsStudentAcademicRecord" sar
    SET 
        "CreditsAttempted" = cc.total_credits,
        "CreditsEarned" = CASE 
            WHEN sar."AcademicStanding" = 'Academic Probation' THEN FLOOR(cc.total_credits * 0.7)
            ELSE FLOOR(cc.total_credits * 0.9)
        END
    FROM credit_count cc
    WHERE sar."StudentAcademicRecordId" = cc."StudentAcademicRecordId";

    -- Ensure financial aid totals make sense
    -- Fix awards to sum to the total granted
    WITH award_sums AS (
        SELECT 
            fa."FinancialAidId",
            fa."AmountGranted",
            COALESCE(SUM(faa."AwardAmount"), 0) AS total_awarded
        FROM "FinancialAid" fa
        LEFT JOIN "FinancialAidAward" faa ON fa."FinancialAidId" = faa."FinancialAidId"
        GROUP BY fa."FinancialAidId", fa."AmountGranted"
    )
    UPDATE "FinancialAidAward" faa
    SET "AwardAmount" = faa."AwardAmount" * (aw.amount_granted / aw.total_awarded)
    FROM (
        SELECT
            as_inner."FinancialAidId",
            as_inner."AmountGranted" as amount_granted,
            as_inner.total_awarded
        FROM award_sums as_inner
        WHERE as_inner.total_awarded > 0 AND as_inner.total_awarded != as_inner."AmountGranted"
    ) AS aw
    WHERE faa."FinancialAidId" = aw."FinancialAidId";

END $$;