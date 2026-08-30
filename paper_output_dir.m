function d = paper_output_dir()
%PAPER_OUTPUT_DIR  Directory the paper artifacts (figures, tables, CSVs) live in.
%
%   The benchmark and plotting scripts write their artifacts here, and two of
%   them read fig_transport_fits_data.mat back from it. Resolution order:
%
%     1. local_paths.m       -- untracked; return an absolute path from it to
%                               point the scripts at a manuscript tree.
%     2. COLLISION_PAPER_DIR -- environment variable, same purpose. Note that
%                               MATLAB started from the macOS Dock does not
%                               inherit a shell profile, so prefer (1) there.
%     3. <repo>/results/figures -- the default on a fresh clone. results/ is
%                               untracked, so nothing lands in the git tree.
%
%   The directory is created if it does not exist.

if exist('local_paths', 'file') == 2
    d = local_paths();
else
    d = getenv('COLLISION_PAPER_DIR');
    if isempty(d)
        d = fullfile(fileparts(mfilename('fullpath')), 'results', 'figures');
    end
end
if ~exist(d, 'dir')
    mkdir(d);
end
end
