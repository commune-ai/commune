 # start of file
import {
  Box,
  Container,
  Heading,
  SimpleGrid,
  Icon,
  Text,
  Stack,
  HStack,
  VStack,
  useColorModeValue,
} from '@chakra-ui/react';
import { CheckIcon } from '@chakra-ui/icons';

// Replace these with actual features of your platform
const features = [
  {
    id: 1,
    title: 'Build Equity From Day One',
    text: 'A portion of each rent payment is converted to ownership tokens, allowing you to build equity immediately.',
  },
  {
    id: 2,
    title: 'No Down Payment Required',
    text: 'Start your homeownership journey without the need for a large upfront down payment.',
  },
  {
    id: 3,
    title: 'Transparent Ownership Tracking',
    text: 'Blockchain technology provides complete transparency in tracking your growing ownership stake.',
  },
  {
    id: 4,
    title: 'Flexible Exit Options',
    text: 'If you need to move, you can sell your accumulated equity tokens or continue building remotely.',
  },
  {
    id: 5,
    title: 'Maintenance Fund Protection',
    text: 'A portion of payments goes to a dedicated maintenance fund, ensuring the property remains in good condition.',
  },
  {
    id: 6,
    title: 'Path to Full Ownership',
    text: 'Continue accumulating equity until you own the property outright or refinance the remaining portion.',
  },
];

export default function FeatureSection() {
  return (
    <Box p={4} bg={useColorModeValue('white', 'gray.800')}>
      <Container maxW={'container.xl'} py={12}>
        <Stack spacing={4} as={Container} maxW={'3xl'} textAlign={'center'}>
          <Heading fontSize={'3xl'}>A Better Way to Own a Home</Heading>
          <Text color={'gray.600'} fontSize={'xl'}>
            Home2Home bridges the gap between renting and owning with innovative
            features designed to make homeownership accessible to everyone.
          </Text>
        </Stack>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={10} mt={10}>
          {features.map((feature) => (
            <HStack key={feature.id} align={'top'}>
              <Box color={'green.400'} px={2}>
                <Icon as={CheckIcon} />
              </Box>
              <VStack align={'start'}>
                <Text fontWeight={600}>{feature.title}</Text>
                <Text color={'gray.600'}>{feature.text}</Text>
              </VStack>
            </HStack>
          ))}
        </SimpleGrid>
      </Container>
    </Box>
  );
}
