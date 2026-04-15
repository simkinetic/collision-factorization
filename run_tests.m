function run_tests()
    % RUN_TESTS  Discovers and runs all unit tests in the 'tests' folder
    
    clc;
    
    % 1. Define Paths relative to this script
    root_dir = fileparts(mfilename('fullpath'));
    src_dir = fullfile(root_dir, 'src');
    test_dir = fullfile(root_dir, 'tests');
    
    % 2. Check folders exist
    if ~exist(src_dir, 'dir')
        error('Source folder not found: %s', src_dir);
    end
    if ~exist(test_dir, 'dir')
        warning('Test folder not found: %s', test_dir);
        return;
    end
    
    % 3. Add paths temporarily (Auto-remove when function exits)
    addpath(src_dir);
    addpath(test_dir);
    
    cleanupObj1 = onCleanup(@() rmpath(src_dir));
    cleanupObj2 = onCleanup(@() rmpath(test_dir));
    
    fprintf('Running tests in: %s\n', test_dir);
    fprintf('--------------------------------------------------\n');
    
    % 4. Run the Test Suite
    try
        import matlab.unittest.TestSuite;
        suite = TestSuite.fromFolder(test_dir);
        results = run(suite);
        
        % Optional: Display a nice summary table
        if ~isempty(results)
            disp(table(results));
        end
        
    catch e
        fprintf(2, 'Error executing tests: %s\n', e.message);
    end
end

