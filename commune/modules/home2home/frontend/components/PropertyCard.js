 # start of file
import NextLink from 'next/link';
import {
  Box,
  Image,
  Badge,
  Text,
  Stack,
  Heading,
  Flex,
  Button,
  useColorModeValue,
} from '@chakra-ui/react';

export default function PropertyCard({ property }) {
  return (
    <Box
      maxW={'445px'}
      w={'full'}
      bg={useColorModeValue('white', 'gray.900')}
      boxShadow={'2xl'}
      rounded={'md'}
      p={6}
      overflow={'hidden'}
      transition="all 0.3s"
      _hover={{
        transform: 'translateY(-5px)',
        boxShadow: 'xl',
      }}
    >
      <Box
        h={'210px'}
        bg={'gray.100'}
        mt={-6}
        mx={-6}
        mb={6}
        pos={'relative'}
      >
        <Image
          src={property.image || '/images/house-placeholder.jpg'}
          alt={property.address}
          objectFit="cover"
          w="full"
          h="full"
        />
      </Box>
      <Stack>
        <Flex justify="space-between" align="center">
          <Text
            color={'blue.500'}
            textTransform={'uppercase'}
            fontWeight={800}
            fontSize={'sm'}
            letterSpacing={1.1}
          >
            Rent-to-Own
          </Text>
          <Flex>
            <Badge colorScheme="blue" mr={1}>
              {property.bedrooms} beds
            </Badge>
            <Badge colorScheme="green">
              {property.bathrooms} baths
            </Badge>
          </Flex>
        </Flex>
        <Heading
          color={useColorModeValue('gray.700', 'white')}
          fontSize={'xl'}
          fontFamily={'body'}
          noOfLines={1}
        >
          {property.address}
        </Heading>
        <Text color={'gray.500'} noOfLines={2}>
          {property.squareFeet.toLocaleString()} sqft • Built in {property.yearBuilt || 'N/A'}
        </Text>
      </Stack>
      <Stack mt={6} direction={'row'} spacing={4} align={'center'}>
        <Stack direction={'column'} spacing={0} fontSize={'sm'}>
          <Text fontWeight={600} fontSize="xl">
            ${property.price.toLocaleString()}
          </Text>
          <Text color={'gray.500'}>Property Value</Text>
        </Stack>
      </Stack>
      <Button
        as={NextLink}
        href={`/property/${property.id}`}
        mt={8}
        w={'full'}
        bg={'blue.500'}
        color={'white'}
        rounded={'md'}
        _hover={{
          transform: 'translateY(-2px)',
          boxShadow: 'lg',
          bg: 'blue.600',
        }}
      >
        View Details
      </Button>
    </Box>
  );
}
