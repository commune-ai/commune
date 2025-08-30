 # start of file
import NextLink from 'next/link';
import {
  Box,
  Container,
  Link,
  SimpleGrid,
  Stack,
  Text,
  Flex,
  useColorModeValue,
  Input,
  IconButton,
  Button,
  Heading,
} from '@chakra-ui/react';
import { FaTwitter, FaInstagram, FaLinkedin, FaGithub } from 'react-icons/fa';
import { ArrowForwardIcon } from '@chakra-ui/icons';

const ListHeader = ({ children }) => {
  return (
    <Text fontWeight={'500'} fontSize={'lg'} mb={2}>
      {children}
    </Text>
  );
};

export default function Footer() {
  return (
    <Box
      bg={useColorModeValue('gray.50', 'gray.900')}
      color={useColorModeValue('gray.700', 'gray.200')}
      borderTop={1}
      borderStyle={'solid'}
      borderColor={useColorModeValue('gray.200', 'gray.900')}
    >
      <Container as={Stack} maxW={'container.xl'} py={10}>
        <SimpleGrid
          templateColumns={{ sm: '1fr 1fr', md: '2fr 1fr 1fr 1fr 2fr' }}
          spacing={8}
        >
          <Stack spacing={6}>
            <Box>
              <Heading as="h3" size="md" color="blue.500">
                Home2Home
              </Heading>
            </Box>
            <Text fontSize={'sm'}>
              Revolutionizing rent-to-own through real estate tokenization.
              Building equity with every payment.
            </Text>
            <Stack direction={'row'} spacing={6}>
              <IconButton
                aria-label={'Twitter'}
                icon={<FaTwitter />}
                rounded={'full'}
                size={'sm'}
              />
              <IconButton
                aria-label={'Instagram'}
                icon={<FaInstagram />}
                rounded={'full'}
                size={'sm'}
              />
              <IconButton
                aria-label={'LinkedIn'}
                icon={<FaLinkedin />}
                rounded={'full'}
                size={'sm'}
              />
              <IconButton
                aria-label={'GitHub'}
                icon={<FaGithub />}
                rounded={'full'}
                size={'sm'}
              />
            </Stack>
          </Stack>
          <Stack align={'flex-start'}>
            <ListHeader>Company</ListHeader>
            <Link as={NextLink} href={'/about'}>About</Link>
            <Link as={NextLink} href={'/careers'}>Careers</Link>
            <Link as={NextLink} href={'/contact'}>Contact</Link>
            <Link as={NextLink} href={'/press'}>Press</Link>
          </Stack>
          <Stack align={'flex-start'}>
            <ListHeader>Resources</ListHeader>
            <Link as={NextLink} href={'/how-it-works'}>How It Works</Link>
            <Link as={NextLink} href={'/faq'}>FAQ</Link>
            <Link as={NextLink} href={'/blog'}>Blog</Link>
            <Link as={NextLink} href={'/testimonials'}>Testimonials</Link>
          </Stack>
          <Stack align={'flex-start'}>
            <ListHeader>Legal</ListHeader>
            <Link as={NextLink} href={'/privacy'}>Privacy Policy</Link>
            <Link as={NextLink} href={'/terms'}>Terms of Service</Link>
            <Link as={NextLink} href={'/cookies'}>Cookie Policy</Link>
            <Link as={NextLink} href={'/licenses'}>Licenses</Link>
          </Stack>
          <Stack align={'flex-start'}>
            <ListHeader>Stay up to date</ListHeader>
            <Stack direction={'row'}>
              <Input
                placeholder={'Your email address'}
                bg={useColorModeValue('white', 'gray.800')}
                border={1}
                borderColor={useColorModeValue('gray.300', 'gray.700')}
                _focus={{
                  bg: 'white',
                  borderColor: 'blue.500',
                }}
              />
              <IconButton
                colorScheme={'blue'}
                aria-label={'Subscribe'}
                icon={<ArrowForwardIcon />}
              />
            </Stack>
          </Stack>
        </SimpleGrid>
      </Container>
      
      <Box
        borderTopWidth={1}
        borderStyle={'solid'}
        borderColor={useColorModeValue('gray.200', 'gray.700')}
      >
        <Container
          as={Stack}
          maxW={'container.xl'}
          py={4}
          direction={{ base: 'column', md: 'row' }}
          spacing={4}
          justify={{ md: 'space-between' }}
          align={{ md: 'center' }}
        >
          <Text>© 2023 Home2Home. All rights reserved</Text>
          <Stack direction={'row'} spacing={6}>
            <Link href={'#'}>Privacy</Link>
            <Link href={'#'}>Terms</Link>
            <Link href={'#'}>Contact</Link>
          </Stack>
        </Container>
      </Box>
    </Box>
  );
}
