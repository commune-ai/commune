# ~/.bashrc

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# Don't put duplicate lines in the history
HISTCONTROL=ignoredups:ignorespace

# Append to the history file, don't overwrite it
shopt -s histappend

# History length
HISTSIZE=1000
HISTFILESIZE=2000

# Check window size after each command
shopt -s checkwinsize

# Make less more friendly for non-text input files
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# Set a fancy prompt
PS1='\[\033[01;32m\]\u@dev\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# Enable color support of ls
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# Some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Python aliases
alias python='python3'
alias pip='pip3'
alias ipy='ipython'

# Development aliases
alias gst='git status'
alias gd='git diff'
alias gc='git commit'
alias gp='git push'

# Set default editor
export EDITOR=vim

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Python development settings
export PYTHONDONTWRITEBYTECODE=1  # Don't create .pyc files
export PYTHONUNBUFFERED=1         # Don't buffer python output