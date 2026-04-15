classdef TestSphericalHarmonics < matlab.unittest.TestCase
    properties
        L_max = 5; % Test with a manageable order
        NZ_Labels
    end

    methods(TestClassSetup)
        function generateLabels(testCase)
            % Generate the labels once for the entire test class
            testCase.NZ_Labels = gaunt_generate_labels(testCase.L_max);
        end
    end

    methods(Test)
        function testTriangleInequality(testCase)
            % Rule: |l1 - l2| <= l3 <= l1 + l2
            for r = 1:size(testCase.NZ_Labels, 1)
                q = double(testCase.NZ_Labels(r, :));
                [l1, ~] = testCase.q2lm(q(1));
                [l2, ~] = testCase.q2lm(q(2));
                [l3, ~] = testCase.q2lm(q(3));

                lower_bound = abs(l1 - l2);
                upper_bound = l1 + l2;

                diagnosticMsg = sprintf('Triangle Rule Failed at row %d: l=[%d,%d,%d]', r, l1, l2, l3);
                testCase.verifyTrue(l3 >= lower_bound && l3 <= upper_bound, diagnosticMsg);
            end
        end

        function testParity(testCase)
            % Rule: l1 + l2 + l3 must be EVEN for a non-zero integral
            for r = 1:size(testCase.NZ_Labels, 1)
                q = double(testCase.NZ_Labels(r, :));
                [l1, ~] = testCase.q2lm(q(1));
                [l2, ~] = testCase.q2lm(q(2));
                [l3, ~] = testCase.q2lm(q(3));

                sum_l = l1 + l2 + l3;
                diagnosticMsg = sprintf('Parity Rule (Odd Sum) Failed at row %d: sum=%d', r, sum_l);
                testCase.verifyEqual(mod(sum_l, 2), 0, diagnosticMsg);
            end
        end

        function testMagneticProjection(testCase)
            % Rule: m1 + m2 = m3 (Conservation of Angular Momentum)
            for r = 1:size(testCase.NZ_Labels, 1)
                q = double(testCase.NZ_Labels(r, :));
                [~, m1] = testCase.q2lm(q(1));
                [~, m2] = testCase.q2lm(q(2));
                [~, m3] = testCase.q2lm(q(3));

                diagnosticMsg = sprintf('Magnetic Projection Failed at row %d: %d + %d ~= %d', r, m1, m2, m3);
                testCase.verifyEqual(m1 + m2, m3, diagnosticMsg);
            end
        end
        
        function testIndexBounds(testCase)
            % Ensure no indices exceed the SH space dimensions
            num_modes = (testCase.L_max + 1)^2;
            testCase.verifyTrue(all(testCase.NZ_Labels(:) <= num_modes), 'Indices exceed SH space size');
            testCase.verifyTrue(all(testCase.NZ_Labels(:) >= 1), 'Indices must be 1-based');
        end
    end

    methods(Static)
        function [l, m] = q2lm(q)
            % Helper to map linear index q back to degree l and order m
            % q = l^2 + l + m + 1
            l = floor(sqrt(q - 1));
            m = (q - 1) - l^2 - l;
        end
    end
end