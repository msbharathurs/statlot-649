import { createClientFromRequest } from 'npm:@base44/sdk@0.8.21';

// Migration function to create Draw records in StatLot 649
// This runs in AppEnhancer context but writes to StatLot 649 via cross-app access

const DRAW_RECORDS = [
  {"draw_number":4078,"n1":9,"n2":16,"n3":17,"n4":20,"n5":34,"n6":38,"additional":18,"sum":134,"odd_count":2,"even_count":4,"low_count":4,"high_count":2,"decade_1":1,"decade_2":3,"decade_3":0,"decade_4":2,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4079,"n1":2,"n2":15,"n3":17,"n4":18,"n5":39,"n6":45,"additional":26,"sum":136,"odd_count":4,"even_count":2,"low_count":4,"high_count":2,"decade_1":1,"decade_2":3,"decade_3":0,"decade_4":1,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4080,"n1":3,"n2":10,"n3":32,"n4":34,"n5":44,"n6":48,"additional":29,"sum":171,"odd_count":1,"even_count":5,"low_count":2,"high_count":4,"decade_1":2,"decade_2":0,"decade_3":0,"decade_4":2,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4081,"n1":5,"n2":9,"n3":15,"n4":28,"n5":46,"n6":48,"additional":8,"sum":151,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":0,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4082,"n1":1,"n2":5,"n3":7,"n4":11,"n5":19,"n6":47,"additional":44,"sum":90,"odd_count":5,"even_count":1,"low_count":6,"high_count":0,"decade_1":3,"decade_2":2,"decade_3":0,"decade_4":0,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4083,"n1":10,"n2":19,"n3":21,"n4":22,"n5":28,"n6":31,"additional":34,"sum":131,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":1,"decade_2":1,"decade_3":3,"decade_4":1,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4084,"n1":2,"n2":5,"n3":25,"n4":26,"n5":29,"n6":30,"additional":42,"sum":117,"odd_count":3,"even_count":3,"low_count":2,"high_count":4,"decade_1":0,"decade_2":0,"decade_3":4,"decade_4":2,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4085,"n1":7,"n2":10,"n3":11,"n4":21,"n5":32,"n6":48,"additional":27,"sum":129,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4086,"n1":3,"n2":7,"n3":38,"n4":41,"n5":44,"n6":49,"additional":20,"sum":182,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":2,"decade_2":0,"decade_3":0,"decade_4":1,"decade_5":3,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4087,"n1":5,"n2":18,"n3":27,"n4":32,"n5":48,"n6":49,"additional":21,"sum":179,"odd_count":3,"even_count":3,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4088,"n1":1,"n2":10,"n3":37,"n4":40,"n5":45,"n6":47,"additional":19,"sum":180,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":2,"decade_2":0,"decade_3":0,"decade_4":2,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4089,"n1":2,"n2":15,"n3":29,"n4":37,"n5":45,"n6":49,"additional":24,"sum":177,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4090,"n1":10,"n2":26,"n3":28,"n4":35,"n5":37,"n6":46,"additional":20,"sum":182,"odd_count":2,"even_count":4,"low_count":1,"high_count":5,"decade_1":1,"decade_2":0,"decade_3":2,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4091,"n1":11,"n2":27,"n3":31,"n4":33,"n5":34,"n6":36,"additional":13,"sum":172,"odd_count":4,"even_count":2,"low_count":1,"high_count":5,"decade_1":0,"decade_2":1,"decade_3":1,"decade_4":4,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4092,"n1":6,"n2":15,"n3":16,"n4":17,"n5":25,"n6":34,"additional":31,"sum":113,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":1,"decade_2":3,"decade_3":1,"decade_4":1,"decade_5":0,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4093,"n1":10,"n2":15,"n3":17,"n4":33,"n5":36,"n6":45,"additional":34,"sum":156,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":1,"decade_2":2,"decade_3":0,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4094,"n1":12,"n2":21,"n3":26,"n4":27,"n5":35,"n6":44,"additional":10,"sum":165,"odd_count":3,"even_count":3,"low_count":2,"high_count":4,"decade_1":0,"decade_2":1,"decade_3":3,"decade_4":1,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4095,"n1":2,"n2":8,"n3":19,"n4":29,"n5":38,"n6":41,"additional":20,"sum":137,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4096,"n1":7,"n2":8,"n3":17,"n4":29,"n5":32,"n6":42,"additional":1,"sum":135,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4097,"n1":2,"n2":5,"n3":10,"n4":12,"n5":14,"n6":37,"additional":17,"sum":80,"odd_count":2,"even_count":4,"low_count":5,"high_count":1,"decade_1":3,"decade_2":2,"decade_3":0,"decade_4":1,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4098,"n1":9,"n2":11,"n3":24,"n4":32,"n5":39,"n6":49,"additional":26,"sum":164,"odd_count":4,"even_count":2,"low_count":3,"high_count":3,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":2,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4099,"n1":2,"n2":14,"n3":16,"n4":21,"n5":36,"n6":47,"additional":1,"sum":136,"odd_count":2,"even_count":4,"low_count":4,"high_count":2,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4100,"n1":7,"n2":19,"n3":20,"n4":21,"n5":22,"n6":29,"additional":37,"sum":118,"odd_count":4,"even_count":2,"low_count":5,"high_count":1,"decade_1":1,"decade_2":2,"decade_3":3,"decade_4":0,"decade_5":0,"consecutive_count":3,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4101,"n1":30,"n2":32,"n3":40,"n4":43,"n5":45,"n6":49,"additional":5,"sum":239,"odd_count":3,"even_count":3,"low_count":0,"high_count":6,"decade_1":0,"decade_2":0,"decade_3":1,"decade_4":2,"decade_5":3,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4102,"n1":2,"n2":15,"n3":28,"n4":39,"n5":42,"n6":44,"additional":5,"sum":170,"odd_count":2,"even_count":4,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4103,"n1":9,"n2":24,"n3":31,"n4":34,"n5":43,"n6":44,"additional":1,"sum":185,"odd_count":3,"even_count":3,"low_count":1,"high_count":5,"decade_1":1,"decade_2":0,"decade_3":1,"decade_4":2,"decade_5":2,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4104,"n1":22,"n2":25,"n3":29,"n4":31,"n5":34,"n6":43,"additional":11,"sum":184,"odd_count":4,"even_count":2,"low_count":0,"high_count":6,"decade_1":0,"decade_2":0,"decade_3":3,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4105,"n1":1,"n2":4,"n3":18,"n4":24,"n5":37,"n6":42,"additional":26,"sum":126,"odd_count":2,"even_count":4,"low_count":4,"high_count":2,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4106,"n1":4,"n2":13,"n3":22,"n4":36,"n5":38,"n6":46,"additional":12,"sum":159,"odd_count":1,"even_count":5,"low_count":3,"high_count":3,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4107,"n1":2,"n2":3,"n3":4,"n4":16,"n5":22,"n6":39,"additional":48,"sum":86,"odd_count":2,"even_count":4,"low_count":5,"high_count":1,"decade_1":3,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":0,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4108,"n1":10,"n2":11,"n3":16,"n4":24,"n5":34,"n6":35,"additional":1,"sum":130,"odd_count":2,"even_count":4,"low_count":4,"high_count":2,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":2,"decade_5":0,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4109,"n1":3,"n2":4,"n3":8,"n4":13,"n5":14,"n6":17,"additional":20,"sum":59,"odd_count":3,"even_count":3,"low_count":6,"high_count":0,"decade_1":3,"decade_2":3,"decade_3":0,"decade_4":0,"decade_5":0,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4110,"n1":8,"n2":12,"n3":21,"n4":38,"n5":40,"n6":43,"additional":25,"sum":162,"odd_count":3,"even_count":3,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":2,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4111,"n1":1,"n2":3,"n3":33,"n4":38,"n5":39,"n6":42,"additional":31,"sum":156,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":2,"decade_2":0,"decade_3":0,"decade_4":3,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4112,"n1":2,"n2":15,"n3":19,"n4":35,"n5":41,"n6":48,"additional":33,"sum":160,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":1,"decade_2":2,"decade_3":0,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4113,"n1":10,"n2":19,"n3":25,"n4":29,"n5":33,"n6":37,"additional":24,"sum":153,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":2,"decade_4":2,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4114,"n1":6,"n2":8,"n3":9,"n4":20,"n5":45,"n6":49,"additional":21,"sum":137,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":3,"decade_2":1,"decade_3":0,"decade_4":0,"decade_5":2,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4115,"n1":8,"n2":15,"n3":22,"n4":24,"n5":43,"n6":47,"additional":44,"sum":159,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":1,"decade_2":1,"decade_3":2,"decade_4":0,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4116,"n1":1,"n2":6,"n3":9,"n4":11,"n5":29,"n6":36,"additional":12,"sum":92,"odd_count":4,"even_count":2,"low_count":4,"high_count":2,"decade_1":3,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4117,"n1":15,"n2":16,"n3":22,"n4":34,"n5":35,"n6":43,"additional":26,"sum":165,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":0,"decade_2":2,"decade_3":1,"decade_4":2,"decade_5":1,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4118,"n1":19,"n2":22,"n3":26,"n4":37,"n5":40,"n6":46,"additional":14,"sum":190,"odd_count":2,"even_count":4,"low_count":2,"high_count":4,"decade_1":0,"decade_2":1,"decade_3":2,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4119,"n1":10,"n2":15,"n3":22,"n4":31,"n5":42,"n6":48,"additional":4,"sum":168,"odd_count":2,"even_count":4,"low_count":3,"high_count":3,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4120,"n1":13,"n2":14,"n3":19,"n4":22,"n5":31,"n6":42,"additional":41,"sum":141,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":0,"decade_2":3,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4121,"n1":5,"n2":31,"n3":33,"n4":34,"n5":38,"n6":46,"additional":39,"sum":187,"odd_count":3,"even_count":3,"low_count":1,"high_count":5,"decade_1":1,"decade_2":0,"decade_3":0,"decade_4":4,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4122,"n1":2,"n2":4,"n3":8,"n4":19,"n5":35,"n6":39,"additional":7,"sum":107,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":3,"decade_2":1,"decade_3":0,"decade_4":2,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4123,"n1":3,"n2":10,"n3":13,"n4":15,"n5":32,"n6":37,"additional":8,"sum":110,"odd_count":4,"even_count":2,"low_count":4,"high_count":2,"decade_1":2,"decade_2":2,"decade_3":0,"decade_4":2,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4124,"n1":7,"n2":14,"n3":17,"n4":18,"n5":31,"n6":38,"additional":46,"sum":125,"odd_count":4,"even_count":2,"low_count":3,"high_count":3,"decade_1":1,"decade_2":3,"decade_3":0,"decade_4":2,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4125,"n1":4,"n2":12,"n3":14,"n4":24,"n5":36,"n6":38,"additional":17,"sum":128,"odd_count":0,"even_count":6,"low_count":4,"high_count":2,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":2,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4126,"n1":1,"n2":5,"n3":31,"n4":34,"n5":38,"n6":45,"additional":21,"sum":154,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":2,"decade_2":0,"decade_3":0,"decade_4":3,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4127,"n1":10,"n2":19,"n3":22,"n4":34,"n5":39,"n6":43,"additional":35,"sum":167,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":1,"decade_2":1,"decade_3":1,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4128,"n1":3,"n2":20,"n3":24,"n4":29,"n5":32,"n6":44,"additional":46,"sum":152,"odd_count":2,"even_count":4,"low_count":3,"high_count":3,"decade_1":1,"decade_2":1,"decade_3":2,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4129,"n1":2,"n2":11,"n3":12,"n4":19,"n5":25,"n6":36,"additional":16,"sum":105,"odd_count":3,"even_count":3,"low_count":4,"high_count":2,"decade_1":1,"decade_2":3,"decade_3":1,"decade_4":1,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4130,"n1":6,"n2":13,"n3":18,"n4":22,"n5":34,"n6":35,"additional":40,"sum":128,"odd_count":2,"even_count":4,"low_count":4,"high_count":2,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":2,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4131,"n1":3,"n2":9,"n3":12,"n4":18,"n5":19,"n6":34,"additional":24,"sum":95,"odd_count":4,"even_count":2,"low_count":5,"high_count":1,"decade_1":2,"decade_2":3,"decade_3":0,"decade_4":1,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4132,"n1":11,"n2":13,"n3":22,"n4":31,"n5":47,"n6":49,"additional":39,"sum":173,"odd_count":5,"even_count":1,"low_count":3,"high_count":3,"decade_1":0,"decade_2":2,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4133,"n1":8,"n2":25,"n3":27,"n4":34,"n5":45,"n6":47,"additional":19,"sum":186,"odd_count":4,"even_count":2,"low_count":1,"high_count":5,"decade_1":1,"decade_2":0,"decade_3":2,"decade_4":1,"decade_5":2,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4134,"n1":6,"n2":8,"n3":17,"n4":28,"n5":32,"n6":46,"additional":16,"sum":137,"odd_count":1,"even_count":5,"low_count":3,"high_count":3,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4135,"n1":2,"n2":10,"n3":24,"n4":35,"n5":45,"n6":49,"additional":37,"sum":165,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":2,"decade_2":0,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4136,"n1":1,"n2":5,"n3":24,"n4":36,"n5":41,"n6":46,"additional":39,"sum":153,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":2,"decade_2":0,"decade_3":1,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4137,"n1":9,"n2":12,"n3":15,"n4":23,"n5":27,"n6":47,"additional":45,"sum":133,"odd_count":4,"even_count":2,"low_count":4,"high_count":2,"decade_1":1,"decade_2":2,"decade_3":2,"decade_4":0,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4138,"n1":6,"n2":11,"n3":20,"n4":28,"n5":33,"n6":43,"additional":16,"sum":141,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4139,"n1":17,"n2":21,"n3":22,"n4":35,"n5":37,"n6":42,"additional":9,"sum":174,"odd_count":4,"even_count":2,"low_count":3,"high_count":3,"decade_1":0,"decade_2":1,"decade_3":2,"decade_4":2,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4140,"n1":2,"n2":14,"n3":15,"n4":30,"n5":31,"n6":43,"additional":27,"sum":135,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4141,"n1":4,"n2":5,"n3":13,"n4":22,"n5":24,"n6":30,"additional":36,"sum":98,"odd_count":2,"even_count":4,"low_count":5,"high_count":1,"decade_1":2,"decade_2":1,"decade_3":3,"decade_4":0,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4142,"n1":3,"n2":8,"n3":15,"n4":28,"n5":37,"n6":43,"additional":49,"sum":134,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":2,"decade_2":1,"decade_3":1,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4143,"n1":2,"n2":4,"n3":22,"n4":24,"n5":30,"n6":33,"additional":49,"sum":115,"odd_count":1,"even_count":5,"low_count":4,"high_count":2,"decade_1":2,"decade_2":0,"decade_3":3,"decade_4":1,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4144,"n1":11,"n2":18,"n3":20,"n4":32,"n5":38,"n6":39,"additional":34,"sum":158,"odd_count":2,"even_count":4,"low_count":3,"high_count":3,"decade_1":0,"decade_2":3,"decade_3":0,"decade_4":3,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4145,"n1":5,"n2":20,"n3":35,"n4":39,"n5":40,"n6":49,"additional":27,"sum":188,"odd_count":4,"even_count":2,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":0,"decade_4":3,"decade_5":1,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4146,"n1":3,"n2":14,"n3":15,"n4":17,"n5":25,"n6":27,"additional":31,"sum":101,"odd_count":5,"even_count":1,"low_count":4,"high_count":2,"decade_1":1,"decade_2":3,"decade_3":2,"decade_4":0,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4147,"n1":1,"n2":9,"n3":16,"n4":18,"n5":35,"n6":43,"additional":12,"sum":122,"odd_count":4,"even_count":2,"low_count":4,"high_count":2,"decade_1":2,"decade_2":2,"decade_3":0,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4148,"n1":16,"n2":32,"n3":34,"n4":35,"n5":36,"n6":41,"additional":14,"sum":194,"odd_count":2,"even_count":4,"low_count":1,"high_count":5,"decade_1":0,"decade_2":1,"decade_3":0,"decade_4":4,"decade_5":1,"consecutive_count":3,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4149,"n1":4,"n2":11,"n3":21,"n4":23,"n5":31,"n6":35,"additional":48,"sum":125,"odd_count":5,"even_count":1,"low_count":4,"high_count":2,"decade_1":1,"decade_2":1,"decade_3":2,"decade_4":2,"decade_5":0,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4150,"n1":6,"n2":22,"n3":27,"n4":32,"n5":37,"n6":44,"additional":19,"sum":168,"odd_count":2,"even_count":4,"low_count":4,"high_count":2,"decade_1":1,"decade_2":0,"decade_3":2,"decade_4":2,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4151,"n1":10,"n2":11,"n3":13,"n4":26,"n5":32,"n6":39,"additional":44,"sum":131,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":1,"decade_2":2,"decade_3":1,"decade_4":2,"decade_5":0,"consecutive_count":1,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4152,"n1":11,"n2":13,"n3":16,"n4":31,"n5":42,"n6":48,"additional":21,"sum":161,"odd_count":3,"even_count":3,"low_count":3,"high_count":3,"decade_1":0,"decade_2":3,"decade_3":0,"decade_4":1,"decade_5":2,"consecutive_count":0,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4153,"n1":4,"n2":19,"n3":40,"n4":41,"n5":46,"n6":47,"additional":20,"sum":197,"odd_count":3,"even_count":3,"low_count":2,"high_count":4,"decade_1":1,"decade_2":1,"decade_3":0,"decade_4":1,"decade_5":3,"consecutive_count":2,"repeat_from_prev":0,"source":"import"},
  {"draw_number":4154,"n1":6,"n2":18,"n3":24,"n4":26,"n5":36,"n6":48,"additional":5,"sum":158,"odd_count":0,"even_count":6,"low_count":3,"high_count":3,"decade_1":1,"decade_2":1,"decade_3":2,"decade_4":1,"decade_5":1,"consecutive_count":0,"repeat_from_prev":0,"source":"import"}
];

Deno.serve(async (req) => {
  try {
    const base44 = createClientFromRequest(req);
    const user = await base44.auth.me();
    if (!user) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json().catch(() => ({}));
    const action = body.action || 'status';

    if (action === 'get_draws') {
      return Response.json({ 
        draws: DRAW_RECORDS,
        count: DRAW_RECORDS.length,
        range: { start: DRAW_RECORDS[0].draw_number, end: DRAW_RECORDS[DRAW_RECORDS.length-1].draw_number }
      });
    }

    return Response.json({ 
      message: 'StatLot Migration Helper',
      available_actions: ['get_draws'],
      draw_count: DRAW_RECORDS.length
    });

  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
});
