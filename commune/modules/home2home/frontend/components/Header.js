 # start of file
import { useState } from 'react';
import NextLink from 'next/link';
import { 
  Box, 
  Flex, 
  Text, 
  Button, 
  Stack, 
  Link, 
  Menu, 
  MenuButton, 
  MenuList, 
  MenuItem,
  IconButton,
  useDisclosure,
  useColorModeValue,
  HStack
} from '@chakra-ui/react';
import { HamburgerIcon, CloseIcon, ChevronDownIcon } from '@chakra-ui/icons';
import { useWeb3 } from '../hooks/useWeb3';

const NAV_ITEMS = [
  {
    label: 'Properties',
    href: '/properties',
  },
  {
    label: 'How It Works',
    href: '/how-it-works',
  },
  {
    label: 'Resources',
    children: [
      {
        label: 'FAQ',
        subLabel: 'Frequently asked questions',
        href: '/faq',
      },
      {
        label: 'Blog',
        subLabel: 'Latest news and updates',
        href: '/blog',
      },
    ],
  },
  {
    label: 'About',
    href: '/about',
  },
];

export default function Header() {
  const { isOpen, onToggle } = useDisclosure();
  const { isConnected, connect, disconnect, account } = useWeb3();
  
  return (
    <Box>
      <Flex
        bg={useColorModeValue('white', 'gray.800')}
        color={useColorModeValue('gray.600', 'white')}
        minH={'60px'}
        py={{ base: 2 }}
        px={{ base: 4 }}
        borderBottom={1}
        borderStyle={'solid'}
        borderColor={useColorModeValue('gray.200', 'gray.900')}
        align={'center'}
      >
        <Flex
          flex={{ base: 1, md: 'auto' }}
          ml={{ base: -2 }}
          display={{ base: 'flex', md: 'none' }}
        >
          <IconButton
            onClick={onToggle}
            icon={
              isOpen ? <CloseIcon w={3} h={3} /> : <HamburgerIcon w={5} h={5} />
            }
            variant={'ghost'}
            aria-label={'Toggle Navigation'}
          />
        </Flex>
        
        <Flex flex={{ base: 1 }} justify={{ base: 'center', md: 'start' }}>
          <Link
            as={NextLink}
            href="/"
            textAlign={{ base: 'center', md: 'left' }}
            fontFamily={'heading'}
            fontWeight={'bold'}
            color={useColorModeValue('blue.600', 'white')}
            fontSize={'xl'}
          >
            Home2Home
          </Link>

          <Flex display={{ base: 'none', md: 'flex' }} ml={10}>
            <DesktopNav />
          </Flex>
        </Flex>

        <Stack
          flex={{ base: 1, md: 0 }}
          justify={'flex-end'}
          direction={'row'}
          spacing={6}
        >
          {isConnected ? (
            <Menu>
              <MenuButton
                as={Button}
                rounded={'full'}
                variant={'outline'}
                cursor={'pointer'}
                minW={0}
                rightIcon={<ChevronDownIcon />}
              >
                {account ? `${account.substring(0, 6)}...${account.substring(account.length - 4)}` : 'Connected'}
              </MenuButton>
              <MenuList>
                <MenuItem as={NextLink} href="/dashboard">Dashboard</MenuItem>
                <MenuItem as={NextLink} href="/profile">Profile</MenuItem>
                <MenuItem onClick={disconnect}>Disconnect</MenuItem>
              </MenuList>
            </Menu>
          ) : (
            <Button
              display={{ base: 'none', md: 'inline-flex' }}
              fontSize={'sm'}
              fontWeight={600}
              color={'white'}
              bg={'blue.500'}
              onClick={connect}
              _hover={{
                bg: 'blue.600',
              }}
            >
              Connect Wallet
            </Button>
          )}
        </Stack>
      </Flex>

      <MobileNav isOpen={isOpen} />
    </Box>
  );
}

const DesktopNav = () => {
  const linkColor = useColorModeValue('gray.600', 'gray.200');
  const linkHoverColor = useColorModeValue('blue.800', 'white');
  const popoverContentBgColor = useColorModeValue('white', 'gray.800');

  return (
    <HStack spacing={4}>
      {NAV_ITEMS.map((navItem) => (
        <Box key={navItem.label}>
          {navItem.children ? (
            <Menu>
              <MenuButton
                as={Link}
                p={2}
                fontSize={'sm'}
                fontWeight={500}
                color={linkColor}
                _hover={{
                  textDecoration: 'none',
                  color: linkHoverColor,
                }}
              >
                {navItem.label} <ChevronDownIcon />
              </MenuButton>
              <MenuList bg={popoverContentBgColor}>
                {navItem.children.map((child) => (
                  <MenuItem 
                    key={child.label} 
                    as={NextLink}
                    href={child.href}
                  >
                    {child.label}
                  </MenuItem>
                ))}
              </MenuList>
            </Menu>
          ) : (
            <Link
              as={NextLink}
              href={navItem.href ?? '#'}
              p={2}
              fontSize={'sm'}
              fontWeight={500}
              color={linkColor}
              _hover={{
                textDecoration: 'none',
                color: linkHoverColor,
              }}
            >
              {navItem.label}
            </Link>
          )}
        </Box>
      ))}
    </HStack>
  );
};

const MobileNav = ({ isOpen }) => {
  return (
    <Box
      display={{ base: isOpen ? 'block' : 'none', md: 'none' }}
      p={4}
      bg={useColorModeValue('white', 'gray.800')}
      borderBottom={1}
      borderStyle={'solid'}
      borderColor={useColorModeValue('gray.200', 'gray.900')}
    >
      <Stack as={'nav'} spacing={4}>
        {NAV_ITEMS.map((navItem) => (
          <MobileNavItem key={navItem.label} {...navItem} />
        ))}
      </Stack>
    </Box>
  );
};

const MobileNavItem = ({ label, children, href }) => {
  const { isOpen, onToggle } = useDisclosure();

  return (
    <Stack spacing={4}>
      <Flex
        py={2}
        as={Link}
        href={href ?? '#'}
        justify={'space-between'}
        align={'center'}
        _hover={{
          textDecoration: 'none',
        }}
        onClick={children && onToggle}
      >
        <Text
          fontWeight={600}
          color={useColorModeValue('gray.600', 'gray.200')}
        >
          {label}
        </Text>
        {children && (
          <ChevronDownIcon
            w={6}
            h={6}
            transform={isOpen ? 'rotate(180deg)' : ''}
            transition={'all .25s ease-in-out'}
          />
        )}
      </Flex>

      {children && (
        <Stack
          mt={2}
          pl={4}
          borderLeft={1}
          borderStyle={'solid'}
          borderColor={useColorModeValue('gray.200', 'gray.700')}
          display={isOpen ? 'block' : 'none'}
        >
          {children.map((child) => (
            <Link
              key={child.label}
              py={2}
              href={child.href}
            >
              {child.label}
            </Link>
          ))}
        </Stack>
      )}
    </Stack>
  );
};
