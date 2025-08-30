" Basic settings
set nocompatible              " Use Vim defaults
set bs=indent,eol,start      " Allow backspacing over everything
set history=50               " Keep 50 lines of command history
set ruler                    " Show cursor position
set number                   " Show line numbers
set autoindent              " Auto-indent new lines
set smartindent             " Enable smart-indent
set smarttab                " Enable smart-tabs
set expandtab               " Use spaces instead of tabs
set softtabstop=4           " Number of spaces per Tab
set shiftwidth=4            " Number of auto-indent spaces
set incsearch               " Show matches while typing
set hlsearch                " Highlight all matches

" Enable syntax highlighting
syntax on
filetype plugin indent on

" Colors
set background=dark
highlight Comment ctermfg=green

" Python specific settings
autocmd FileType python set colorcolumn=80
autocmd FileType python set textwidth=79
autocmd FileType python set formatoptions+=t
autocmd FileType python set fileformat=unix

" Status line
set laststatus=2
set statusline=%F%m%r%h%w\ [FORMAT=%{&ff}]\ [TYPE=%Y]\ [POS=%l,%v][%p%%]

" Key mappings
map <C-n> :NERDTreeToggle<CR>
inoremap jk <ESC>

" Better command-line completion
set wildmenu
set wildmode=list:longest

" Show matching brackets
set showmatch

" Search settings
set ignorecase
set smartcase

" No backup files
set nobackup
set nowritebackup
set noswapfile