
"""Tools for running tasks locally or in a cluster environment."""

import abc
import os
import stat
import shutil
import time
import subprocess
import math

# Shell environment variable to specify which environment to use."
AMDTK_PARALLEL_ENV = 'AMDTK_PARALLEL_ENV'

# Possible environment. If the AMDTK_PARALLEL_ENV is not set then the
# first of the list ("local") will be taken.
ENVS = [
    "local",
    "sge",
    "openccs"
    "openccs-single"
    ]

# Template for the script to run. Beware before changing this script:
# AMDTK recipes relies on this specific processing so be careful.
JOB_TEMPLATE = """{header}

echo "Job started on $(date)."
echo "Hostname: $(hostname)."

cd {cwd}
source {profile}



LIST={flist}

s=$(( (JOB_ID-1)*{njobs} + 1 ))
e=$(( (JOB_ID)*{njobs} ))
if [ "$e" -gt "{total}" ]; then
    e={total}
fi

if [ "$s" -le "{total}" ]; then
    LINES=$(sed -n "${{s}},${{e}}p" $LIST)
    for LINE in $LINES ; do
        ITEM1=$(echo $LINE | awk -F ':' '{{print $1}}')
        ITEM2=$(echo $LINE | awk -F ':' '{{print $2}}')

        echo running: {cmd}
        {cmd}
        retcode=$?

        if [ $retcode -ne 0 ]; then
            echo 1 "{qdir}/{name}.$JOB_ID.log" > {qdir}/{name}."$JOB_ID".status
            exit 1
        fi
    done
fi

echo 0 "{qdir}/{name}.$JOB_ID.log" > {qdir}/{name}."$JOB_ID".status

echo Job ended on $(date)
"""


# Template for the python file_list_worker. Beware before changing this
# script: AMDTK recipes relies on this specific processing so be careful.
PYTHON_JOB_TEMPLATE = """{header}

echo "Job started on $(date)."
echo "Hostname: $(hostname)."

cd {cwd}
source {profile}

ITEM1={{0}}
ITEM2={{1}}

file_list_worker {flist} {njobs} ${{JOB_ID}} {cmd}
retcode=$?

if [ $retcode -ne 0 ]; then
    echo 1 "{qdir}/{name}.$JOB_ID.log" > {qdir}/{name}."$JOB_ID".status
    exit 1
fi

echo 0 "{qdir}/{name}.$JOB_ID.log" > {qdir}/{name}."$JOB_ID".status

echo Job ended on $(date)
"""


class AmdtkUnknownParallelEnvironment(Exception):
    """If the user has specified an unknown environment."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ParallelEnv(metaclass=abc.ABCMeta):
    """Abstract base class for all parallel environments.

    Implements also some processing common for all environments.

    Methods
    -------
    getEnvironment()
        Get an :class:`ParallelEnv` object according to the user
        settings.
    header()
        Header of the script file specific to the parallel environment.
    prepareOutputDirectory(outdir)
        Prepare the output directory of the command.
    computeTaskLoad(list_items, ntasks)
        Compute the work load of each task.
    prepareScript(name, cmd, options, profile)
        Final preparation of the script.
    waitJobs()
        Wait for all the jobs to finish.
    run()
        Run all the tasks in the parallel environment.
    """

    @staticmethod
    def getEnvironment():
        """Get an :class:`ParallelEnv` object according to the user
        settings.

        """
        try:
            env = os.environ[AMDTK_PARALLEL_ENV]
        except KeyError:
            env = ENVS[0]

        if env not in ENVS:
            raise AmdtkUnknownParallelEnvironment('unknown parallel '
                                                  'environment: ' + str(env))

        if env == ENVS[0]:
            return LocalParallelEnv()
        elif env == ENVS[2]:
            return OpenCCSParallelEnv()
        elif env == ENVS[3]:
            return OpenCCSSemiParallelEnv()
        else:
            return SGEParallelEnv()

    @abc.abstractmethod
    def header(self):
        """Header of the script file specific to the parallel
        environment.

        """
        pass

    def prepareOutputDirectory(self, outdir):
        """Prepare the output directory of the command.

        Parameters
        ----------
        outdir: string
            Output directory of the command. A specific directory will
            be created inside "outdir" to store data related to the
            parallel environment and the execution of the script.

        """
        self.outdir = os.path.abspath(outdir)
        self.qdir = os.path.join(self.outdir, 'parallel')
        if os.path.exists(self.qdir):
            shutil.rmtree(self.qdir)
        os.makedirs(self.qdir)

    def computeTaskLoad(self, list_items, ntasks):
        """Compute the work load of each task.

        Parameters
        ----------
        list_items: string
            Path to list file that contains the information for each
            job to run.
        ntasks: int
            Number of task (i.e. processes) to use.

        """
        self.list_items = os.path.abspath(list_items)
        with open(list_items, 'r') as f:
            lines = f.readlines()
        self.total = len(lines)
        self.ntasks = min(ntasks, self.total)
        self.njobs = math.ceil(self.total / self.ntasks)

    def prepareScript(self, name, cmd, options, profile, python_script=False):
        """Final preparation of the script.

        Parameters
        ----------
        name: string
            Some informative name about the command.
        cmd: string
            a shell command to execute in parallel.
        options: string
            Possible options for the parallel environment.
        profile: string
            Path to the profile file (i.e. shell script that will be
            executed before running the  command).
        python_script: bolean
            Treat command as python script to be run with python file
            list worker. Script needs to be importable and have a main
            function accepting one agument in the form of sys.argv.

        """
        self.script_path = os.path.join(self.qdir, name+'.task')
        self.name = name
        data = {
            'name': name,
            'qdir': self.qdir,
            'options': options,
            'njobs': self.njobs,
            'flist': self.list_items,
            'outdir': self.outdir,
            'cwd': os.getcwd(),
            'cmd': cmd,
            'profile': profile,
            'total': self.total,
            'ntasks': self.ntasks,
            'pid': os.getpid()
        }

        data['header'] = self.header().format(**data)

        with open(self.script_path, 'w') as f:
            if python_script == True:
                f.write(PYTHON_JOB_TEMPLATE.format(**data))
            else:
                f.write(JOB_TEMPLATE.format(**data))
            st = os.stat(self.script_path)
            os.chmod(self.script_path, st.st_mode | stat.S_IEXEC)

    def waitJobs(self):
        """Wait for all the jobs to finish."""

        wait = True
        while wait:
            wait = False
            for i in range(self.ntasks):
                status_file = self.name + '.' + str(i+1) + '.status'
                if not os.path.exists(os.path.join(self.qdir, status_file)):
                    wait = True
            time.sleep(1)

        retval = 0
        for i in range(self.ntasks):
            status_file = self.name + '.' + str(i+1) + '.status'
            with open(os.path.join(self.qdir, status_file), 'r') as f:
                tokens = f.readline().strip().split()
                errcode = int(tokens[0])
                logfile = tokens[1]
                if errcode != 0:
                    retval += 1
                    print('Job has failed. See', logfile)
        return retval

    @abc.abstractmethod
    def run(self):
        """Run all the tasks in the parallel environment."""
        pass


class LocalParallelEnv(ParallelEnv):
    LOCAL_HEADER = """
cd {cwd}
JOB_ID=$1
"""

    def header(self):
        """Header of the script file specific to the parallel
        environment.

        """
        return self.LOCAL_HEADER

    def run(self):
        """Run all the tasks in the (local) parallel environment."""

        to_close = []
        for i in range(self.ntasks):
            logfilename = \
                os.path.join(self.qdir, self.name + '.' + str(i+1) + '.log')
            logfile = open(logfilename, 'w')
            to_close.append(logfile)
            shell = os.environ['SHELL']
            cmd_list = [shell, self.script_path, str(i+1)]
            subprocess.Popen(cmd_list, stdout=logfile, stderr=logfile)
        failed = self.waitJobs()
        for f in to_close:
            f.close()

        return failed


class OpenCCSSemiParallelEnv(ParallelEnv):
    CCS_HEADER = """#!/bin/bash
#CCS {options}
#CCS --cwd={cwd}

JOB_ID=$1
"""

    def header(self):
        """Header of the script file specific to the parallel
        environment.

        """
        return self.CCS_HEADER

    def run(self):
        """Run all the tasks in the (openccs) parallel environment."""

        try:
            to_close = []
            for i in range(self.ntasks):
                logfilename = \
                    os.path.join(self.qdir, self.name + '.' + str(i+1) + '.log')
                logfile = open(logfilename, 'w')
                to_close.append(logfile)
                cmd_list = ['ccsalloc',
                            '--output={}'.format(logfilename),
                            '--stderr={}'.format(logfilename),
                            '--name={}-{}-{}'.format(self.name, os.getpid(), i+1),
                            self.script_path,
                            str(i+1)]
                process = subprocess.Popen(cmd_list, stdout=logfile, stderr=logfile)
                process.wait()
            failed = self.waitJobs()
            for f in to_close:
                f.close()
        except KeyboardInterrupt:
            for i in range(self.ntasks):
                cmd_list = ['ccskill',
                            '{}-{}-{}'.format(self.name, os.getpid(), i+1)]
                process = subprocess.Popen(cmd_list)
                process.wait()
            exit(1)
        return failed


class SGEParallelEnv(ParallelEnv):
    SGE_HEADER = """
#$ -S /bin/bash
#$ -N {name}
#$ -j y
#$ -o {qdir}/{name}."$TASK_ID".log
#$ -V
#$ {options}
#$ -cwd
#$ -t 1-{ntasks}

JOB_ID="$SGE_TASK_ID"

"""

    def header(self):
        """Header of the script file specific to the parallel
        environment.

        """
        return self.SGE_HEADER

    def run(self):
        """Run all the tasks in the (sge) parallel environment."""

        qcmd = 'qsub < ' + self.script_path
        try:
            subprocess.call(qcmd, shell=True, stdout=subprocess.PIPE)
            failed = self.waitJobs()
        except KeyboardInterrupt:
            delcmd = 'qdel ' + self.name
            subprocess.call(delcmd, shell=True)
            exit(1)
        return failed


class OpenCCSParallelEnv(ParallelEnv):
    CCS_HEADER = """#! /bin/bash
#CCS -N {name}-{pid}
#CCS -o {qdir}/{name}.%a.log
#CCS --stderr={qdir}/{name}.%a.log
#CCS {options}
#CCS --cwd={cwd}
#CCS -J 1-{ntasks}

JOB_ID="$CCS_ARRAY_INDEX"

"""

    def header(self):
        """Header of the script file specific to the parallel
        environment.

        """
        return self.CCS_HEADER

    def run(self):
        """Run all the tasks in the (openccs) parallel environment."""

        qcmd = 'ccsalloc ' + self.script_path
        try:
            subprocess.call(qcmd, shell=True, stdout=subprocess.PIPE)
            failed = self.waitJobs()
        except KeyboardInterrupt:
            delcmd = 'ccskill {}-{}'.format(self.name, os.getpid())
            subprocess.call(delcmd, shell=True)
            exit(1)
        return failed
