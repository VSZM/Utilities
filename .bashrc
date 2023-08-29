########################
# AUTO STARTIN SSH AGENT 
########################

env=~/.ssh/agent.env

agent_load_env () { test -f "$env" && . "$env" >| /dev/null ; }

agent_start () {
    (umask 077; ssh-agent >| "$env")
    . "$env" >| /dev/null ; }

agent_load_env

# agent_run_state: 0=agent running w/ key; 1=agent w/o key; 2= agent not running
agent_run_state=$(ssh-add -l >| /dev/null 2>&1; echo $?)

if [ ! "$SSH_AUTH_SOCK" ] || [ $agent_run_state = 2 ]; then
    agent_start
    ssh-add
elif [ "$SSH_AUTH_SOCK" ] && [ $agent_run_state = 1 ]; then
    ssh-add
fi

unset env

#############################
# AUTO STARTING SSH AGENT DONE
#############################


function nonzero_return() {
	RETVAL=$?
	[ $RETVAL -ne 0 ] && echo "$RETVAL"
}


function get_bizottsagi_anyag() {
    #ftp://vb:v328G1@/vb/2021/04_20/anyag.zip
    echo open 195.70.44.107 >> ftp &echo user vb v328G1 >> ftp &echo binary >> ftp &echo get 2021/04_20/anyag.zip >> ftp &echo bye >> ftp &ftp -n -v -s:ftp &del ftp
}

export PS1="\[\e[32m\]\u\[\e[m\]@\[\e[33m\]\h\[\e[m\] \[\e[31m\]\w\[\e[m\][\`nonzero_return\`]\n\\$> "

export LESSOPEN="| /usr/bin/src-hilite-lesspipe.sh %s"
export LESS=" -R "
export ENV=local
export DISPLAY=localhost:0
export QHOME="/c/tools/q"


# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=50000
HISTFILESIZE=500000
PROMPT_COMMAND='history -a'


#https://www.cyberciti.biz/tips/bash-aliases-mac-centos-linux-unix.html

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    alias python="python3"
    alias ls="ls --color=auto"
    alias cp="gcp -i"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    alias python="python3"
    alias ls="ls -G"
    alias cp="rsync -r --progress"
else
    alias python="winpty python"
    alias ls="ls --color=auto"
fi

alias ll="ls -lAh"
alias lsdir="ll -d */"
alias grep="grep --color=auto"
alias mkdir="mkdir -pv"
alias path="echo -e ${PATH//:/\\n}"
alias histgrep="history | grep"
alias psgrep="ps -ef | grep"
alias sshpuboff="mv ~/.ssh/id_rsa ~/.ssh/id_rsa.backup && mv ~/.ssh/id_rsa.pub ~/.ssh/id_rsa.pub.backup && mv ~/.ssh/config ~/.ssh/config.backup && echo 'ssh public key authentication turned off'"
alias sshpubon="mv ~/.ssh/id_rsa.backup ~/.ssh/id_rsa && mv ~/.ssh/id_rsa.pub.backup ~/.ssh/id_rsa.pub && mv ~/.ssh/config.backup ~/.ssh/config && echo 'ssh public key authentication turned on'"
alias rmdir="rm -rf"
alias mvnverify="mvn clean verify -Dskip.integration.tests=false -Dmaven.test.failure.ignore=false"
alias git-list-large-objects="git rev-list --objects --all \
| git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
| sed -n 's/^blob //p' \
| sort --numeric-sort --key=2 \
| cut -c 1-12,41- \
| $(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest"
alias refresh_bash="source ~/.bash_profile"
alias fixendline="perl -pi -e 's/\r\n/\n/g'"
alias unixendline_folder="find . -type f -exec dos2unix {} \;"

# confirmation #
alias mv="mv -i"
alias ln="ln -i"

# cycle through options with tab
bind TAB:menu-complete

# git auto complete
source ~/git-completion.bash

# machine specific
alias q='QHOME=/c/tools/q/ /c/tools/q/w64/q.exe'
alias start_google_proxy="(cd /c/tools/google_cloud_proxy/ && ./cloud_sql_proxy_x64.exe -instances=amethyst-111111:europe-west4:database-2=tcp:54321) &"
alias start_pgadmin="'/c/Program Files/PostgreSQL/13/pgAdmin 4/bin/pgAdmin4.exe' &"
alias cdp="cd ~/Tresorit/Programming/Python && conda activate ds"
alias ssh_dsd_srv="ssh gabor.szegedi@157.181.176.110 -p 10025"
alias ssh_hp="ssh -XY vszm@192.168.1.192"
alias ssh_momo_gcloud="ssh -i /c/Users/VSZM5/.ssh/google_compute_engine VSZM5@35.204.18.223"
alias cdmomolytics="cd /q/Nextcloud/Tresorit/Politika/Data_Science_Team/"

if hash conda 2>/dev/null; then
    conda init bash
fi

echo "Your environment is setup and ready to go `whoami`!"