function d = paper_output_dir()
%PAPER_OUTPUT_DIR  The directory every benchmark and plotting script writes to.
%
%   Always <repo>/results. Created on first use. Nothing is written outside the
%   repository, and nothing in results/ is tracked except the shipped seed data
%   named in the README.

d = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(d, 'dir')
    mkdir(d);
end
end
