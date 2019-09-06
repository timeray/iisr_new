# Script to copy output files from remote server
import stat
import paramiko
import datetime as dt
from pathlib import Path
from getpass import getpass
from iisr.data_manager import DataManager
from contextlib import contextmanager


def is_dir(path_: Path, sftp_client_: paramiko.SFTPClient):
    return stat.S_ISDIR(sftp_client_.stat(str(path_)).st_mode)


def recursive_list_filepaths(root_dirpath: Path, sftp_client_: paramiko.SFTPClient):
    print('Go to {}'.format(str(root_dirpath)))
    sftp_client_.chdir(str(root_dirpath))
    for name in sftp_client_.listdir():
        if name in ['.', '..']:
            continue

        new_path = root_dirpath / name
        if is_dir(new_path, sftp_client_):
            yield from recursive_list_filepaths(new_path, sftp_client_)
        else:
            yield Path(new_path)


@contextmanager
def safe_sftp(ssh_client_: paramiko.SSHClient):
    sftp_client_ = ssh_client_.open_sftp()  # type: paramiko.SFTPClient
    yield sftp_client_
    sftp_client_.close()


username = 'setov'
server_name = 'hind'

if server_name == 'hind':
    hostname = '10.0.0.83'
    port = paramiko.config.SSH_PORT
elif server_name == 'orda':
    hostname = '81.18.122.84'
    port = 2222
else:
    raise ValueError('Incorrect server name')

ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname=hostname, username=username, password=getpass(), port=port)

copy_dates = [dt.date(2017, 7, 1) + dt.timedelta(days=i) for i in range(31)]
# copy_dates = [dt.date(2019, 5, 12)]

remote_results_dir = Path('/') / 'home' / 'users' / username / 'iisr_new' / 'output'
match_suffix = '.png'

with safe_sftp(ssh_client) as sftp_client:  # type: paramiko.SFTPClient
    n_copied = 0
    for date in copy_dates:
        remote_manager = DataManager(main_folder_path=remote_results_dir, create_main_folder=False)
        local_manager = DataManager()

        remote_root_dirpath = remote_manager.get_figures_folder_path(date=date,
                                                                     create_new_folders=False)

        try:
            for remote_filepath in recursive_list_filepaths(remote_root_dirpath, sftp_client):
                if str(remote_filepath).endswith(match_suffix):
                    rel_path = remote_filepath.relative_to(remote_root_dirpath)
                    local_filepath = local_manager.get_figures_folder_path(date=date) / rel_path

                    if not local_filepath.parent.exists():
                        local_filepath.parent.mkdir(parents=True)

                    n_copied += 1
                    print('Copy {} to {}'.format(str(remote_filepath), str(local_filepath)))
                    sftp_client.get(str(remote_filepath), str(local_filepath))
        except FileNotFoundError:
            print(f'Cannot find path {remote_root_dirpath}')

    print(f'Done: copied {n_copied} files')
